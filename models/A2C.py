import gymnasium as gym
import torch
import numpy as np
from helper_functions import evaluate_policy_continuous

def train_model(
        model_id: str,
        env_id,
        solved_threshold: float,
        success_threshold: float,
        test_freq: int = 10,
        n_workers:int = 4,
        n_steps: int = 500,
        model_policy=None,
        model_value=None,
        entropy_coef: float = 0.05,
        self_train_itr: int = 100_000,
        model_policy_learning_rate: float = 0.0001,
        model_value_learning_rate: float = 0.001,
        gamma: float = 0.99,
        lamda: float = 0.95,
        verbose: bool = True,
):
    envs = gym.make_vec(env_id, num_envs=n_workers, vectorization_mode="sync")
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    action_high = torch.tensor(envs.single_action_space.high, dtype=torch.float32)

    # stats tracker
    losses = []
    total_rewards = []
    env_steps = 0

    optimizer_policy = torch.optim.RMSprop(model_policy.parameters(), lr=model_policy_learning_rate)
    optimizer_value = torch.optim.RMSprop(model_value.parameters(), lr=model_value_learning_rate)

    # state shape: (n_workers, obs_dim)
    states, _ = envs.reset()

    for i in range(self_train_itr):

        # TD(lambda) style collection
        episode = []
        env_n_steps = 0
        for _ in range(n_steps):
            # Gymnasium VectorEnvs auto-reset individual workers automatically no need for checks
            with torch.no_grad():
                '''
                # Get V(s) and (mean, std) for Gaussian distribution
                means, stds = model_policy(torch.tensor(states, dtype=torch.float32))
                '''
                raw_means, stds = model_policy(torch.tensor(states, dtype=torch.float32))
                means = torch.tanh(raw_means) * action_high
                # create Gaussian
                dists = torch.distributions.Normal(loc=means, scale=stds)
                # Sample to get action
                actions = dists.sample()

            # avoids illegal actions
            clipped_actions = actions.clamp(envs.single_action_space.low[0],envs.single_action_space.high[0])
            next_states, rewards, terminated, truncated, infos = envs.step(clipped_actions.detach().numpy())

            # Save step and continue
            episode.append((states, actions, rewards, terminated))
            states = next_states
            env_steps += 1
            env_n_steps += 1

        # determine if we need bootstrap
        with torch.no_grad():
            all_states = torch.tensor(
                np.array([s for s, _, _, _ in episode]), dtype=torch.float32
            )  # shape: (T, n_workers, obs_dim)

            # V(s)
            values = model_value(all_states.view(-1, state_dim)).view(env_n_steps, n_workers)

            last_states = torch.tensor(states, dtype=torch.float32)
            last_dones = torch.tensor(episode[-1][3], dtype=torch.float32)

            # V(s'), if is terminal then don't bootstrap
            next_value = model_value(last_states).squeeze(-1) * (1 - last_dones)

        gae_collection = torch.zeros(env_n_steps, n_workers)
        gae = torch.zeros(n_workers)

        for t in reversed(range(env_n_steps)):
            t_states, t_actions, t_rewards, t_dones = episode[t]
            t_rewards = torch.tensor(t_rewards, dtype=torch.float32)
            t_dones = torch.tensor(t_dones, dtype=torch.float32)

            # r + gamma * V(s) - V(s')
            # Need to check both V(s) and V(s') because of vectorized env
            delta = t_rewards + gamma * next_value * (1 - t_dones) - values[t].squeeze(-1)

            # delta + gamma * delta
            gae = delta + gamma * lamda * (1 - t_dones) * gae
            gae_collection[t] = gae
            next_value = values[t].squeeze(-1)


        # Update our model
        all_states = torch.tensor(np.array([s for s, _, _, _ in episode]), dtype=torch.float32)
        all_actions = torch.stack([a for _, a, _, _ in episode])

        ''''# get means and std to make multiple dists
        means, stds = model_policy(all_states.view(-1, state_dim))'''

        '''means = means.view(env_n_steps, n_workers, action_dim)'''
        raw_means, stds = model_policy(all_states.view(-1, state_dim))
        scaled_means = torch.tanh(raw_means) * action_high
        means = scaled_means.view(env_n_steps, n_workers, action_dim)
        stds = stds.view(env_n_steps, n_workers, action_dim)

        # V(s)
        value_preds = model_value(all_states.view(-1, state_dim)).squeeze(-1)
        value_preds = value_preds.view(env_n_steps, n_workers)

        dists = torch.distributions.Normal(means, stds)
        entropy = dists.entropy().sum(-1)  # shape (T, n_workers)
        weighted_entropy = entropy * entropy_coef
        log_probs = dists.log_prob(all_actions).sum(-1)  # shape (T, n_workers)

        advantages = gae_collection.detach()
        # normalize advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # clamping to avoid grad explosion
        advantages = torch.clamp(advantages, -2.0, 2.0)
        loss_policy = -(advantages * log_probs + weighted_entropy).mean()

        # Reconstruct the target Return (R = A + V) or (R = gae + V(s)) derived from A = R - V(s)
        returns = gae_collection + values.detach()
        # try l1 instead of MSE
        loss_value = torch.nn.functional.smooth_l1_loss(value_preds, returns)

        optimizer_policy.zero_grad()
        loss_policy.backward()
        torch.nn.utils.clip_grad_norm_(model_policy.parameters(), max_norm=1.0)
        optimizer_policy.step()

        optimizer_value.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model_value.parameters(), max_norm=1.0)
        optimizer_value.step()

        if i % test_freq == 0:
            if verbose: print(f"Env step: {env_steps}")
            avg_reward_per_ep = evaluate_policy_continuous(
                model_policy,
                solved_threshold=solved_threshold,
                success_threshold=success_threshold,
                env_id=env_id,
                n_episodes=20,
                verbose=verbose,
            )
            total_rewards.append(avg_reward_per_ep)

            if avg_reward_per_ep >= success_threshold:
                torch.save(
                    model_policy.state_dict(),
                    f"checkpoint models/{model_id}_{env_id}_checkpoint_{avg_reward_per_ep:.0f}.pth"
                )

            if avg_reward_per_ep >= solved_threshold:
                torch.save(model_policy.state_dict(), f"solved models/{model_id}_{env_id}_A2C_solved.pth")
                print("Solved! Model saved.")
                return losses, total_rewards

    return losses, total_rewards










