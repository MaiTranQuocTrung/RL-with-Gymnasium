import gymnasium as gym
import numpy as np
import torch
from helper_functions import evaluate_policy_discrete, make_env, evaluate_policy_continuous_on_policy
from gymnasium.wrappers import NormalizeObservation, TransformObservation

# noinspection PyUnboundLocalVariable
def train_model(
        model_id : str,
        env_id,
        solved_threshold: float,
        success_threshold: float,
        model = None,
        self_train_itr: int = 100_000,
        model_learning_rate: float = 0.0005,
        gamma: float = 0.99,
        verbose: bool = True,
):
    env = make_env(env_id)

    # stats tracker
    losses = []
    total_rewards = []
    env_steps = 0

    # neural net variables
    optimizer = torch.optim.RMSprop(model.parameters(), lr=model_learning_rate)

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

            # probability distribution of actions (probability of taking an action at a given state)
            probs = model(torch.tensor(state, dtype=torch.float32))

            # create a distribution, meaning idx binded to probabilities
            dist = torch.distributions.Categorical(probs=probs)

            # This is a weighted sample which mimics Softmax Exploration
            action = dist.sample().squeeze().item()

            next_state, reward, terminated, truncated, info = env.step(action)
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


        # calculate the total loss and optimize
        total_loss = 0

        for (state, action, _), G in zip(episode, returns):

            probs = model(torch.tensor(state, dtype=torch.float32))
            dist = torch.distributions.Categorical(probs=probs)

            """
                log_prob(action): how confident the policy was in choosing this action
                G: how good the rest of the episode turned out after taking this action
                log_prob * G = "how confident we were" * "how good it turned out"
                we want to maximize our confidence with a high G, meaning we want to minimize total_loss hence the "-"
            """

            total_loss += -dist.log_prob(torch.tensor(action, dtype=torch.int64)) * G

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

            eval_env = gym.make(env_id)
            eval_env = NormalizeObservation(eval_env)
            eval_env = TransformObservation(eval_env, lambda obs: np.clip(obs, -10.0, 10.0), eval_env.observation_space)

            avg_reward_per_episodes = evaluate_policy_continuous_on_policy(model, solved_threshold=solved_threshold,
                                                                           success_threshold=success_threshold,
                                                                           env=eval_env, verbose=verbose, n_episodes=20)

            total_rewards.append(avg_reward_per_episodes)

            if avg_reward_per_episodes >= solved_threshold:
                print(f"\nSolved in {i + 1} iterations!")

                torch.save(
                    model.state_dict(),
                    f"solved models/{model_id}_{env_id}_REINFORCE_solved.pth"
                )

                return losses, total_rewards

    return losses, total_rewards




