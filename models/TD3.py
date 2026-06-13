import torch
import torch.nn as nn
import numpy as np
import random
from helper_functions import make_env, evaluate_policy_continous_off_policy, polyak_update
from collections import deque
import copy
import gymnasium as gym


def train_model(
                model_actor,
                model_critic,
                model_second_critic,
                env_id,
                solved_threshold: float,
                success_threshold: float,
                actor_sync_freq: int = 2,
                test_freq: int = 1,
                sigma: float = 0.2,
                self_train_itr: int = 200,
                actor_learning_rate: float = 0.0003,
                critic_learning_rate: float = 0.0003,
                gamma: float = 0.99,
                buffer_size: int = 50_000,
                min_sample: int = 320,
                batch_size: int = 256,
                verbose = True
                ):
    # create a new env to build our dataset
    env = make_env(env_id)()

    # stats tracker
    losses = []
    total_rewards = []


    loss_fn = nn.MSELoss()
    actor_optimizer = torch.optim.Adam(model_actor.parameters(), lr=actor_learning_rate)
    critic_optimizer = torch.optim.Adam(model_critic.parameters(), lr=critic_learning_rate)
    critic_second_optimizer = torch.optim.Adam(model_second_critic.parameters(), lr=critic_learning_rate)

    # define target network
    target_actor = copy.deepcopy(model_actor)
    target_critic = copy.deepcopy(model_critic)
    target_critic_two = copy.deepcopy(model_second_critic)

    # define replay buffer
    replay_buffer = deque(maxlen=buffer_size)

    total_steps = 0
    for i in range (self_train_itr):
        # sample of env
        terminated = truncated = False
        state, _ = env.reset()

        #frac = 1.0 - (i / self_train_itr)
        #current_sigma = max(frac * sigma, sigma * 0.1)

        current_sigma = sigma

        while not (terminated or truncated):
            #loc = mean, scale = std
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            raw_action = model_actor(state_tensor).squeeze(0).detach().cpu().numpy()
            noise = np.random.normal(0, current_sigma, size=raw_action.shape)
            action = raw_action + noise

            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, terminated, truncated, info = env.step(action)
            replay_buffer.append((state, action, reward, next_state, float(terminated)))

            state = next_state
            total_steps += 1

            # we can start sampling from buffer when it is sufficiently big
            if len(replay_buffer) > min_sample:
                # sample min_sample from replay buffer
                states, actions, rewards, next_states, is_failures = zip(*random.sample(replay_buffer, batch_size))

                # convert to tensors to input into our model
                states = torch.from_numpy(np.array(states)).float()
                actions = torch.from_numpy(np.array(actions)).float()
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.from_numpy(np.array(next_states)).float()
                is_failures = torch.tensor(is_failures, dtype=torch.float32)

                # Target Q-values
                with torch.no_grad():
                    target_actions = target_actor(next_states)
                    # We are trying to build Q(s',a'), in TD3 gaussian noise is also injected here
                    noise = torch.normal(mean=0, std=0.2, size=target_actions.shape, device=next_states.device)
                    noise = noise.clamp(-0.5, 0.5)

                    next_actions = (target_actions + noise).clamp(
                        min=torch.tensor(env.action_space.low).float().to(next_states.device),
                        max=torch.tensor(env.action_space.high).float().to(next_states.device)
                    )

                    # now we get Q(s',a') => use twin network
                    q_next_critic_one = target_critic(next_states, next_actions).squeeze(-1)
                    q_next_critic_two = target_critic_two(next_states, next_actions).squeeze(-1)
                    q_next = torch.min(q_next_critic_two, q_next_critic_one)
                    # build the target
                    q_target_critic = rewards + gamma * q_next * (1 - is_failures)


                # training critic
                q_critic = model_critic(states, actions).squeeze(-1)
                loss_critic = loss_fn(q_critic, q_target_critic)
                critic_optimizer.zero_grad()
                loss_critic.backward()
                torch.nn.utils.clip_grad_norm_(model_critic.parameters(), max_norm=10)
                critic_optimizer.step()

                # training second critic
                q_critic = model_second_critic(states, actions).squeeze(-1)
                loss_critic = loss_fn(q_critic, q_target_critic)
                critic_second_optimizer.zero_grad()
                loss_critic.backward()
                torch.nn.utils.clip_grad_norm_(model_second_critic.parameters(), max_norm=10)
                critic_second_optimizer.step()

                if total_steps % actor_sync_freq == 0:
                    # training actor
                    predicted_actions = model_actor(states)
                    actor_q_values = model_critic(states, predicted_actions)
                    loss_actor = -actor_q_values.mean()
                    actor_optimizer.zero_grad()
                    loss_actor.backward()
                    torch.nn.utils.clip_grad_norm_(model_actor.parameters(), max_norm=10)
                    actor_optimizer.step()

                    # sync network
                    polyak_update(model_critic, target_critic, tau=0.005)
                    polyak_update(model_second_critic, target_critic_two, tau=0.005)
                    polyak_update(model_actor, target_actor, tau=0.005)


        eval_env = gym.make(env_id)


        if verbose and i % test_freq == 0:
            print(f"Self-train itr: {i} | Env step: {total_steps}")
            avg_reward_per_episodes = evaluate_policy_continous_off_policy( model_actor,
                                                                            env=eval_env,
                                                                            solved_threshold=solved_threshold,
                                                                            success_threshold=success_threshold,
                                                                            verbose=verbose
                                                                            )
            total_rewards.append(avg_reward_per_episodes)

            if avg_reward_per_episodes >= solved_threshold:
                print(f"\nSolved in {i + 1} iterations!")
                torch.save(model_actor.state_dict(), f"solved models/{env_id}_TD3_solved.pth")
                return losses, total_rewards


    return losses, total_rewards



