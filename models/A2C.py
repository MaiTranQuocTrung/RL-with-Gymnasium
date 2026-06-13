import gymnasium as gym
import torch
import numpy as np
from helper_functions import evaluate_policy_continuous_on_policy, setup_envs
from gymnasium.wrappers import NormalizeObservation, TransformObservation

def train_model(
        env_id,
        solved_threshold: float,
        success_threshold: float,
        test_freq: int = 10,
        n_workers:int = 4,
        n_steps: int = 30,
        model_policy=None,
        model_value=None,
        entropy_coef: float = 0.01,
        self_train_itr: int = 100_000,
        model_policy_learning_rate: float = 0.0001,
        model_value_learning_rate: float = 0.001,
        gamma: float = 0.99,
        lamda: float = 0.95,
        verbose: bool = True,
):
    envs = setup_envs(env_id, n_workers, gamma)

    states, _ = envs.reset()

    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    # stats tracker
    losses = []
    total_rewards = []
    env_steps = 0

    optimizer_policy = torch.optim.Adam(model_policy.parameters(), lr=model_policy_learning_rate)
    optimizer_value = torch.optim.Adam(model_value.parameters(), lr=model_value_learning_rate)

    # state shape: (n_workers, obs_dim)
    states, _ = envs.reset()


    for i in range(self_train_itr):
        # linear decay lr, start at base lr to 10% of lr towards the end
        frac = 1.0 - (i / self_train_itr)
        for param_group in optimizer_policy.param_groups:
            param_group["lr"] = max(frac * model_policy_learning_rate, model_policy_learning_rate * 0.1)
        for param_group in optimizer_value.param_groups:
            param_group["lr"] = max(frac * model_value_learning_rate, model_value_learning_rate * 0.1)

        # do the same with entropy
        current_entropy_coef = max(frac * entropy_coef, entropy_coef * 0.1)

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
                means, stds = model_policy(torch.tensor(states, dtype=torch.float32))
                # create Gaussian
                dists = torch.distributions.Normal(loc=means, scale=stds)
                # Sample to get action
                actions = dists.sample()

            # avoids illegal actions
            #clipped_actions = actions.clamp(envs.single_action_space.low[0],envs.single_action_space.high[0])
            next_states, rewards, terminated, truncated, infos = envs.step(actions.detach().numpy())

            # because at 1000 steps the agent teleports back to the start with a reward of 0 before terminating/truncating
            # just before the truncation happens we must assign the true reward value to it and treat truncation as termination
            # this means no bootstrapping. This is logical because truncation also teleports the agent back to the start
            if "final_observation" in infos:
                for w in range(n_workers):
                    # Only bootstrap if it hit the 1,000 step limit (truncated)
                    if infos["_final_observation"][w] and truncated[w]:
                        # Get the true state BEFORE the auto-reset
                        final_obs = infos["final_observation"][w]
                        final_obs_tensor = torch.tensor(final_obs, dtype=torch.float32).unsqueeze(0)

                        # Predict V(s) for the true final state
                        terminal_value = model_value(final_obs_tensor).squeeze(-1).item()

                        # Fold it directly into the reward for this step
                        rewards[w] += gamma * terminal_value

            # Save step and continue
            episode.append((states, actions, rewards, np.logical_or(terminated, truncated)))
            states = next_states
            env_steps += n_workers
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

        # get means and std to make multiple dists
        means, stds = model_policy(all_states.view(-1, state_dim))
        means = means.view(env_n_steps, n_workers, action_dim)
        stds = stds.view(env_n_steps, n_workers, action_dim)

        # V(s)
        value_preds = model_value(all_states.view(-1, state_dim)).squeeze(-1)
        value_preds = value_preds.view(env_n_steps, n_workers)

        dists = torch.distributions.Normal(means, stds)
        entropy = dists.entropy().sum(-1)  # shape (T, n_workers)
        weighted_entropy = entropy * current_entropy_coef
        log_probs = dists.log_prob(all_actions).sum(-1)  # shape (T, n_workers)

        advantages = gae_collection.detach()
        loss_policy = -(advantages * log_probs + weighted_entropy).mean()

        # Reconstruct the target Return (R = A + V) or (R = gae + V(s)) derived from A = R - V(s)
        returns = gae_collection + values.detach()
        loss_value = torch.nn.functional.mse_loss(value_preds, returns)

        optimizer_policy.zero_grad()
        loss_policy.backward()
        torch.nn.utils.clip_grad_norm_(model_policy.parameters(), max_norm=1.0)
        optimizer_policy.step()

        optimizer_value.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model_value.parameters(), max_norm=1.0)
        optimizer_value.step()

        if i % test_freq == 0:
            if verbose: print(f"Self-train itr: {i} | Env step: {env_steps}")

            eval_env = gym.make(env_id)
            eval_env = NormalizeObservation(eval_env)
            eval_env = TransformObservation(eval_env, lambda obs: np.clip(obs, -10.0, 10.0), eval_env.observation_space)

            train_env_unwrapped = envs
            while not hasattr(train_env_unwrapped, 'obs_rms') and hasattr(train_env_unwrapped, 'env'):
                train_env_unwrapped = train_env_unwrapped.env

            eval_env_unwrapped = eval_env
            while not hasattr(eval_env_unwrapped, 'obs_rms') and hasattr(eval_env_unwrapped, 'env'):
                eval_env_unwrapped = eval_env_unwrapped.env

            if hasattr(train_env_unwrapped, 'obs_rms') and hasattr(eval_env_unwrapped, 'obs_rms'):
                eval_env_unwrapped.obs_rms.mean = train_env_unwrapped.obs_rms.mean.copy()
                eval_env_unwrapped.obs_rms.var = train_env_unwrapped.obs_rms.var.copy()
                eval_env_unwrapped.obs_rms.count = train_env_unwrapped.obs_rms.count

            avg_reward_per_ep = evaluate_policy_continuous_on_policy(model_policy, solved_threshold=solved_threshold,
                                                                     success_threshold=success_threshold, env=eval_env,
                                                                     verbose=verbose, n_episodes=20)
            total_rewards.append(avg_reward_per_ep)
            '''
            if avg_reward_per_ep >= success_threshold:
                torch.save(
                    model_policy.state_dict(),
                    f"checkpoint models/{model_id}_{env_id}_checkpoint_{avg_reward_per_ep:.0f}.pth"
                )
            '''
            if avg_reward_per_ep >= solved_threshold:
                torch.save(model_policy.state_dict(), f"solved models/{env_id}_A2C_solved.pth")
                np.save(f"solved dist stats/A2C_{env_id}_mean.npy", eval_env_unwrapped.obs_rms.mean)
                np.save(f"solved dist stats/A2C_{env_id}_var.npy", eval_env_unwrapped.obs_rms.var)
                print("Solved! Model saved.")
                return losses, total_rewards

    return losses, total_rewards










