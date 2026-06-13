import gymnasium as gym
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from helper_functions import evaluate_policy_continuous_on_policy, setup_envs
from gymnasium.wrappers import NormalizeObservation, TransformObservation


def train_model(
        env_id,
        solved_threshold: float,
        success_threshold: float,
        n_epochs: int = 10,
        mini_batch_size: int = 500,
        clip_coef: float = 0.2,
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
                # log probs, use sum(-1) to calculate the cumulative log probs for that set of actions
                old_log_probs = dists.log_prob(actions).sum(-1).detach()
                # V(s) from value network
                old_values = model_value(torch.tensor(states, dtype=torch.float32)).squeeze(-1).detach()

            next_states, rewards, terminated, truncated, infos = envs.step(actions.detach().numpy())

            # because at 1000 steps the agent teleports back to the start with a reward of 0 before terminating/truncating
            # just before the truncation happens we must assign the true reward value to it and treat truncation as termination
            # this means no bootstrapping. This is logical because truncation also teleports the agent back to the start
            if "final_observation" in infos:
                for w in range(n_workers):
                    if infos["_final_observation"][w] and truncated[w]:
                        final_obs = infos["final_observation"][w]

                        if hasattr(envs, 'obs_rms'):
                            mean = envs.obs_rms.mean
                            var = envs.obs_rms.var
                            # Standardize: (x - mean) / sqrt(var + epsilon)
                            final_obs = (final_obs - mean) / np.sqrt(var + 1e-8)
                            final_obs = np.clip(final_obs, -10.0, 10.0)

                        final_obs_tensor = torch.tensor(final_obs, dtype=torch.float32).unsqueeze(0)
                        terminal_value = model_value(final_obs_tensor).squeeze(-1).item()

                        rewards[w] += gamma * terminal_value

            # Save step and continue
            episode.append((states, actions, rewards, np.logical_or(terminated, truncated), old_log_probs, old_values))
            states = next_states
            env_steps += n_workers
            env_n_steps += 1

        # determine if we need bootstrap
        with torch.no_grad():
            all_states = torch.tensor(
                np.array([s for s, _, _, _, _, _ in episode]), dtype=torch.float32
            )  # shape: (T, n_workers, obs_dim)

            # V(s)
            values = model_value(all_states.view(-1, state_dim)).view(env_n_steps, n_workers)

            # the last state is states after the while loop exits
            last_states = torch.tensor(states, dtype=torch.float32)
            # check if the last state is really termina;
            last_dones = torch.tensor(episode[-1][3], dtype=torch.float32)

            # V(s'), if is terminal then don't bootstrap
            next_value = model_value(last_states).squeeze(-1) * (1 - last_dones)

        gae_collection = torch.zeros(env_n_steps, n_workers)
        gae = torch.zeros(n_workers)

        for t in reversed(range(env_n_steps)):
            t_states, t_actions, t_rewards, t_dones, _, _ = episode[t]
            t_rewards = torch.tensor(t_rewards, dtype=torch.float32)
            t_dones = torch.tensor(t_dones, dtype=torch.float32)

            # r + gamma * V(s) - V(s')
            # Need to check both V(s) and V(s') because of vectorized env
            delta = t_rewards + gamma * next_value * (1 - t_dones) - values[t].squeeze(-1)

            # A_t = δ_t + γλ * A_{t+1}
            gae = delta + gamma * lamda * (1 - t_dones) * gae
            gae_collection[t] = gae
            next_value = values[t].squeeze(-1)

        # calculate advantage
        advantages = gae_collection.detach()
        # normalize advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # clamping to avoid grad explosion
        advantages = torch.clamp(advantages, -2.0, 2.0)

        # Reconstruct the target Return (R = A + V) or (R = gae + V(s)) derived from A = R - V(s)
        # Note that since GAE is only an advantage but we want the actual return which
        # we can use Monte Carlo which will be Gt or use the R = A + V formula
        returns = gae_collection + values.detach()

        all_states = torch.tensor(np.array([s for s, _, _, _, _, _ in episode]), dtype=torch.float32).view(-1,state_dim)
        all_actions = torch.stack([a for _, a, _, _, _, _ in episode]).view(-1, action_dim)
        old_log_probs = torch.stack([old_log_prob for _, _, _, _, old_log_prob, _ in episode]).view(-1)
        old_values = torch.stack([old_val for _, _, _, _, _, old_val in episode]).view(-1)

        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)

        # use mini batch
        dataset = TensorDataset(all_states, all_actions, old_log_probs, old_values, flat_advantages, flat_returns)

        # PPO style training using multiple epochs
        for j in range (n_epochs):
            dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
            for batch in dataloader:
                # 3. Unpack the exact mini-batch instantly
                mb_states, mb_actions, mb_old_log_probs, mb_old_values, mb_advantages, mb_returns = batch

                # make current normal dist
                means_current, stds_current = model_policy(mb_states)
                dists_current = torch.distributions.Normal(means_current, stds_current)

                # entropy calculations, meaning how random or distribution is
                entropy = dists_current.entropy().sum(-1)  # shape (T, n_workers)
                weighted_entropy = entropy * current_entropy_coef
                current_log_probs = dists_current.log_prob(mb_actions).sum(-1)  # shape (T, n_workers)

                # calculate ratio between new and old action log probs, how much better or worse
                # the network thinks an action is
                # ratio = exp(new - old)
                # this is the same as new prob/old pob
                ratio = torch.exp(current_log_probs - mb_old_log_probs)

                # calculate clipped and non clipped "advantages * log_probs"
                # we add entropy as a way to increase randomness
                non_clipped_sug = mb_advantages * ratio
                clipped_sug = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_advantages
                loss_policy = -(torch.min(clipped_sug, non_clipped_sug) + weighted_entropy).mean()

                # V(s)
                value_preds = model_value(mb_states).squeeze(-1)

                # calculate clipped and non clipped error
                value_loss_unclipped = torch.nn.functional.mse_loss(value_preds, mb_returns, reduction='none')
                # Force the new prediction to stay within 20% of the old prediction
                value_clipped = mb_old_values + torch.clamp(value_preds - mb_old_values, -clip_coef, clip_coef)
                value_loss_clipped = torch.nn.functional.mse_loss(value_clipped, mb_returns, reduction='none')

                loss_value = torch.max(value_loss_unclipped, value_loss_clipped).mean()

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
            print(f"Value loss: {loss_value.item():.4f} | Policy loss: {loss_policy.item():.4f}")

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

            if avg_reward_per_ep >= solved_threshold:
                torch.save(model_policy.state_dict(), f"solved models/{env_id}_PPO_solved.pth")
                np.save(f"solved dist stats/PPO_{env_id}_mean.npy", eval_env_unwrapped.obs_rms.mean)
                np.save(f"solved dist stats/PPO_{env_id}_var.npy", eval_env_unwrapped.obs_rms.var)

                print("Solved! Model saved.")
                return losses, total_rewards

    return losses, total_rewards


