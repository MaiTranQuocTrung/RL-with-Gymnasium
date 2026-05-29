import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from helper_functions import get_all_q, evaluate_policy

def train_model(
                model,
                model_id: str,
                solved_threshold: float,
                success_threshold: float,
                env_id,
                self_train_itr: int = 40,
                dataset_size: int = 2000,
                model_learning_rate: float = 0.0005,
                batch_size: int = 128,
                K: int = 40,
                gamma: float = 0.99,
                epsilon: float = 0.5,
                verbose = False,
                ):

    total_env_step = 0

    # create a new env to build our dataset
    env_collect = gym.make(env_id)

    # stats tracker
    losses = []
    total_rewards = []

    # these are our variables for state and action space
    state_space = env_collect.observation_space.shape[0] # number of state features


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=model_learning_rate)

    # how many times are we repeating the collection + training
    for i in range(self_train_itr):
        if verbose:
            print(f"Model: {model_id}, Self training itr: {i + 1}")

        state, _ = env_collect.reset()
        print(f"EPS: {epsilon}")


        #Define our arrays to the following tuple in MDP (s, s', a, r, terminate)
        X_raw = np.zeros((dataset_size, state_space), dtype=np.float32) # state is unique because it stores a tuple of 4 floats
        X_next_raw = np.zeros((dataset_size, state_space), dtype=np.float32) # state is unique because it stores a tuple of 4 floats
        a_raw = np.zeros(dataset_size, dtype=np.int64) # just an int
        r_raw = np.zeros(dataset_size, dtype=np.float32) # just a float
        d_raw = np.zeros(dataset_size, dtype=np.float32) # just a float

        for step in range(dataset_size):
          # eps greedy policy for exploitation vs exploration in RL
            if random.random() < epsilon:
                action = env_collect.action_space.sample()
            else:
                action = np.argmax(get_all_q(state, model))

            # get information about next state
            next_state, reward, isTerminate, truncated, info = env_collect.step(action)

            # this is crucial, if the episode is terminated because of time it is NOT a failure, it is a BIG sucess
            is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
            is_failure = isTerminate and not is_truncated

            # Append values to our MDP tuple
            X_raw[step] = state
            X_next_raw[step] = next_state
            a_raw[step] = action
            r_raw[step] = reward
            d_raw[step] = float(is_failure)

            total_env_step+=1
            # if we haven't collected enough data, reset the env
            if isTerminate or truncated:
                state, _ = env_collect.reset()
            else:
                state = next_state

        # pre-convert to tensors
        X_tensor = torch.tensor(X_raw, dtype=torch.float32)
        X_next_tensor = torch.tensor(X_next_raw, dtype=torch.float32)
        a_tensor = torch.tensor(a_raw, dtype=torch.long)
        r_tensor = torch.tensor(r_raw, dtype=torch.float32)
        d_tensor = torch.tensor(d_raw, dtype=torch.float32)


        # this is the fitted step of NFQ, K is more of less just the epoch
        for k in range(K):

            # Recalculate targets in one batched forward pass!!! This is the main source of instability
            model.eval()
            with torch.no_grad():
                q_next = model(X_next_tensor)               # (2000, action_space) => Q(s', a) for next state actions
                max_q_next = q_next.max(dim=1)[0]            # (2000,) => get the biggest Q(s', a) value
                q_target = r_tensor + gamma * max_q_next * (1.0 - d_tensor) # Update using Q-learning formula, notice 1.0 - d_tensor, this is a nice way of not using if else


            # Train on fresh targets
            model.train()

            # Notice shuffle here, this is extremely important
            loader = DataLoader(TensorDataset(X_tensor, a_tensor, q_target), batch_size=batch_size, shuffle=True)

            k_loss = 0
            for X_batch, a_tensor_batch, y_batch in loader:
                q_values = model(X_batch)
                # specifically the action chosen by the agent
                q_selected = q_values.gather(1, a_tensor_batch.unsqueeze(1)).squeeze(1)
                loss = loss_fn(q_selected, y_batch)
                optimizer.zero_grad()
                loss.backward()
                # avoids gradient explosion which is quite common in RL algorithms, not necessary but good to just have
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                k_loss += loss.item()

            losses.append(k_loss / len(loader))

        avg_reward_per_episodes = evaluate_policy(model, solved_threshold=solved_threshold, success_threshold=success_threshold, env_id=env_id, verbose=True, record=False)
        total_rewards.append(avg_reward_per_episodes)
        print(f"Total env step: {total_env_step}")

        # If the model already has optimal policy don't explore more, that's not needed, just more noise
        if avg_reward_per_episodes >= solved_threshold:
            print(f"\nSolved in {i + 1} iterations!")
            torch.save(model.state_dict(), f"solved models/{model_id}_NFQ_solved.pth")
            return losses, total_rewards

    return losses, total_rewards