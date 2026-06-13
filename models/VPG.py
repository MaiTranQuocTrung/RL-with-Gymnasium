import gymnasium as gym
import numpy as np
import torch
from helper_functions import evaluate_policy_discrete, evaluate_policy_continuous_on_policy, make_env
from gymnasium.wrappers import NormalizeObservation, TransformObservation

# noinspection PyUnboundLocalVariable
def train_model(
        model_id : str,
        env_id,
        solved_threshold: float,
        success_threshold: float,
        model_policy_discrete = None,
        model_policy = None,
        model_value = None,
        self_train_itr: int = 100_000,
        model_learning_rate: float = 0.0005,
        gamma: float = 0.99,
        verbose: bool = True,
        is_discrete: bool = True,
):
    env = make_env(env_id)

    # stats tracker
    losses = []
    total_rewards = []
    env_steps = 0

    # neural net variables
    if is_discrete:
        optimizer_policy = torch.optim.RMSprop(model_policy_discrete.parameters(), lr=model_learning_rate)
    else:
        optimizer_policy = torch.optim.RMSprop(model_policy.parameters(), lr=model_learning_rate)

    optimizer_value = torch.optim.RMSprop(model_value.parameters(), lr=model_learning_rate)

    # collection and training
    for i in range(self_train_itr):

        if verbose:
            print(f"Model: {model_id}, Self training itr: {i + 1}")
            print(f"Env steps: {env_steps}")

        # Monte carlo style collection
        episode = []

        state, _ = env.reset()
        terminated = truncated = False

        while not(terminated or truncated):

            if is_discrete:
                # probability distribution of actions (probability of taking an action at a given state)
                probs = model_policy_discrete(torch.tensor(state, dtype=torch.float32))

                # create a distribution, meaning idx binded to probabilities
                dist = torch.distributions.Categorical(probs=probs)

                # This is a weighted sample which mimics Softmax Exploration
                action = dist.sample().squeeze().item()

                next_state, reward, terminated, truncated, info = env.step(action)

            else:
                # get std and mean for Gaussian dist
                mean, std = model_policy(torch.tensor(state, dtype=torch.float32))

                # create Gaussian
                dist = torch.distributions.Normal(loc=mean, scale=std)

                # Sample to get action
                action = dist.sample()

                # avoids illegal actions
                clipped_action = action.clamp(env.action_space.low[0],env.action_space.high[0])

                next_state, reward, terminated, truncated, info = env.step(clipped_action.detach().numpy())

            episode.append((state, action, reward))

            state = next_state
            env_steps += 1

        # calculate discounted returns
        returns = []
        G = 0.0

        for _, _, reward in reversed(episode):
            G = G * gamma + reward
            returns.append(G)

        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32)

        if is_discrete:

            # calculate the total loss and optimize
            total_loss_policy = 0
            total_loss_value = 0

            for (state, action, _), G in zip(episode, returns):

                probs = model_policy_discrete(torch.tensor(state, dtype=torch.float32))
                dist = torch.distributions.Categorical(probs=probs)

                # get V(s)
                value_state = model_value(torch.tensor(state, dtype=torch.float32))

                # get entropy
                entropy = dist.entropy()
                weighted_entropy = 0.01 * entropy

                # Advantage: detach value so gradients don't flow into value net here
                advantage = (G - value_state).detach()

                """
                    log_prob(action): how confident the policy was in choosing this action
                    G: how good the rest of the episode turned out after taking this action
                    log_prob * G = "how confident we were" * "how good it turned out"
                    we want to maximize our confidence with a high G, meaning we want to minimize total_loss hence the "-"
                """
                total_loss_policy += -(advantage * dist.log_prob(torch.tensor(action, dtype=torch.int64)) + weighted_entropy)
                total_loss_value += (G - value_state)**2

            loss_policy = total_loss_policy / len(episode)
            loss_value = total_loss_value / len(episode)

            optimizer_policy.zero_grad()
            loss_policy.backward()
            optimizer_policy.step()

            optimizer_value.zero_grad()
            loss_value.backward()
            optimizer_value.step()

            avg_reward_per_episodes = evaluate_policy_discrete(model_policy_discrete, solved_threshold=solved_threshold,
                                                               success_threshold=success_threshold, env_id=env_id,
                                                               verbose=True)

            total_rewards.append(avg_reward_per_episodes)

            losses.append((loss_policy.item() + loss_value.item()) / 2)

            if avg_reward_per_episodes >= solved_threshold:
                print(f"\nSolved in {i + 1} iterations!")
                torch.save(model_policy_discrete.state_dict(), f"solved models/{model_id}_{env_id}_VPG_solved.pth")
                return losses, total_rewards

        if not is_discrete:

            # accumulate total loss for backward prop
            total_loss_policy_network = 0
            total_loss_value_network = 0

            for (state, action, _), G in zip(episode, returns):

                # Get V(s) and (mean, std) for Gaussian distribution
                mean, std = model_policy(torch.tensor(state, dtype=torch.float32))

                value_state = model_value(torch.tensor(state, dtype=torch.float32))

                dist = torch.distributions.Normal(loc=mean, scale=std)


                entropy = dist.entropy().sum()
                weighted_entropy = 0.01 * entropy.mean()

                # Compute loss for both networks
                advantage = (G - value_state).detach() # avoid gradiant flow in value model

                total_loss_value_network += (G - value_state) ** 2

                total_loss_policy_network += -(advantage *dist.log_prob(action).sum() + weighted_entropy)

            loss_policy = total_loss_policy_network / len(episode)
            loss_value = total_loss_value_network / len(episode)

            optimizer_policy.zero_grad()
            loss_policy.backward()
            optimizer_policy.step()

            optimizer_value.zero_grad()
            loss_value.backward()
            optimizer_value.step()

            # create normalized env
            eval_env = gym.make(env_id)
            eval_env = NormalizeObservation(eval_env)
            eval_env = TransformObservation(eval_env, lambda obs: np.clip(obs, -10.0, 10.0), eval_env.observation_space)

            avg_reward_per_episodes = evaluate_policy_continuous_on_policy(model_policy,
                                                                           solved_threshold=solved_threshold,
                                                                           success_threshold=success_threshold,
                                                                           env=eval_env, verbose=verbose, n_episodes=20)

            total_rewards.append(avg_reward_per_episodes)

            losses.append((loss_policy.item() + loss_value.item()) / 2)

            if avg_reward_per_episodes >= solved_threshold:
                print(f"\nSolved in {i + 1} iterations!")
                torch.save( model_policy.state_dict(),f"solved models/{model_id}_{env_id}_VPG_solved.pth")
                return losses, total_rewards

    return losses, total_rewards




