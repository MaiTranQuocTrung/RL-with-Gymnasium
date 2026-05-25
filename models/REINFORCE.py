import gymnasium as gym
import torch
from helper_functions import evaluate_policy


def train_model(
        model,
        model_id : str,
        env_id,
        solved_threshold: float,
        success_threshold: float,
        self_train_itr: int = 100_000,
        model_learning_rate: float = 0.0005,
        gamma: float = 0.99,
        verbose: bool = True,
):
    env = gym.make(env_id)

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
            probs = model(torch.tensor(state))
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

        # calculate the total loss and optimize
        total_loss = torch.tensor(0.0)
        for (state, action, _), G in zip(episode, returns):
            probs = model(torch.tensor(state))
            dist = torch.distributions.Categorical(probs=probs)
            total_loss += -dist.log_prob(torch.tensor(action)) * G

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

        avg_reward_per_episodes = evaluate_policy(model, solved_threshold=solved_threshold, success_threshold=success_threshold, env_id=env_id, verbose=True)
        total_rewards.append(avg_reward_per_episodes)

        if avg_reward_per_episodes >= solved_threshold:
            print(f"\nSolved in {i + 1} iterations!")
            torch.save(model.state_dict(), f"solved models/{model_id}_{env_id}_REINFORCE_solved.pth")
            return losses, total_rewards

    return losses, total_rewards




