import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import random

from torch.utils.data import DataLoader, TensorDataset

from helper_functions import get_all_q, evaluate_policy
from collections import deque
import copy


def train_model(
                model,
                model_id: str,
                env_id,
                self_train_itr: int = 200,
                sync_freq: int = 15,
                model_learning_rate: float = 0.0005,
                gamma: float = 0.99,
                buffer_size: int = 50_000,
                min_sample: int = 320,
                verbose = True
                ):
    saved = False
    # create a new env to build our dataset
    env_collect = gym.make(env_id)

    # stats tracker
    losses = []
    total_rewards = []

    # these are our variables for state and action space
    #state_space = env_collect.observation_space.shape[0]

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=model_learning_rate)

    # define target network
    target_network = copy.deepcopy(model)

    # define replay buffer
    replay_buffer = deque(maxlen=buffer_size)

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = (epsilon_min / epsilon) ** (1 / 5000)

    total_steps = 0
    for i in range (self_train_itr):

        if verbose:
            print(f"Model: {model_id}, Self training itr: {i + 1}")

        state, _ = env_collect.reset()
        print(f"EPS: {epsilon}")

        # sample of env
        terminated = truncated = False

        while not (terminated or truncated):
            # sync target network every 10 self learning itr
            if total_steps % sync_freq == 0:
                target_network = copy.deepcopy(model)
                #print("Sync network") if verbose else None
            total_steps += 1

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            if random.random() < epsilon:
                action = env_collect.action_space.sample()
            else:
                action = np.argmax(get_all_q(state, model))

            next_state, reward, terminated, truncated, info = env_collect.step(action)
            is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
            is_failure = terminated and not is_truncated

            replay_buffer.append((state, action, reward, next_state, float(is_failure)))
            state = next_state


            # we can start sampling from buffer when it is sufficiently big
            if len(replay_buffer) > min_sample:
                # sample min_sample from replay buffer
                states, actions, rewards, next_states, is_failures = zip(*random.sample(replay_buffer, min_sample))

                # convert to tensors to input into our model
                states = torch.from_numpy(np.array(states)).float()
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.from_numpy(np.array(next_states)).float()
                is_failures = torch.tensor(is_failures, dtype=torch.float32)

                # Target Q-values
                with torch.no_grad():
                    q_next = target_network(next_states).max(dim=1)[0]
                    q_target = rewards + gamma * q_next * (1 - is_failures)

                # training stats
                epoch_loss = 0

                # We must take in actions to use in training
                loader = DataLoader(TensorDataset(states, actions, q_target),batch_size=64,shuffle=True)

                # train our network on data
                for X_batch, action_batch, y_batch in loader:
                    q_values = model(X_batch)
                    # the agent picked an action during collection, therefore, q_target is only relevant to that specific q value
                    q_selected = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                    loss = loss_fn(q_selected, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                losses.append(epoch_loss / len(loader))

        print(f"Total env step: {total_steps}")
        avg_reward_per_episodes = evaluate_policy(model, env_id, verbose=True, record=False)
        total_rewards.append(avg_reward_per_episodes)

        if avg_reward_per_episodes >= 500.0 and not saved:
            print(f"\nSolved in {i + 1} iterations!")
            torch.save(model.state_dict(), f"{model_id}_DQN_solved.pth")
            return losses, total_rewards


    return losses, total_rewards



