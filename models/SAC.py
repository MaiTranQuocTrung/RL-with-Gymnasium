import torch
import torch.nn as nn
import numpy as np
import random
from helper_functions import make_env, evaluate_policy_continuous_SAC, polyak_update
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

    #Target entropy is typically -dim(actions)
    target_entropy = -np.prod(env.action_space.shape).item()
    device = next(model_actor.parameters()).device
    log_alpha = torch.zeros(1, requires_grad=True, device=device)

    # Give alpha its own optimizer
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=actor_learning_rate)

    # define target network
    target_critic = copy.deepcopy(model_critic)
    target_critic_two = copy.deepcopy(model_second_critic)

    # define replay buffer
    replay_buffer = deque(maxlen=buffer_size)

    total_steps = 0
    for i in range (self_train_itr):
        # sample of env
        terminated = truncated = False
        state, _ = env.reset()

        while not (terminated or truncated):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # SAC uses the actor model similar to policy gradient algs no need for noise insertion
            mean, std = model_actor(state_tensor)
            # Create Gaussian distribution
            dist = torch.distributions.Normal(mean, std)
            # sample from that dist
            raw_action = dist.sample()
            action = torch.tanh(raw_action)
            # need to switch back to numpy for gymnasium
            env_action = action.squeeze(0).cpu().numpy()
            next_state, reward, terminated, truncated, info = env.step(env_action)
            replay_buffer.append((state, env_action, reward, next_state, float(terminated)))

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
                    next_mean, next_std = model_actor(next_states)
                    next_dist = torch.distributions.Normal(next_mean, next_std)

                    raw_next_actions = next_dist.sample()
                    squashed_next_actions = torch.tanh(raw_next_actions)

                    # use twin network to calculate the Q value
                    q_next_critic_one = target_critic(next_states, squashed_next_actions).squeeze(-1)
                    q_next_critic_two = target_critic_two(next_states, squashed_next_actions).squeeze(-1)
                    q_next = torch.min(q_next_critic_two, q_next_critic_one)

                    # Jacobian correction
                    next_log_prob = next_dist.log_prob(raw_next_actions)
                    next_log_prob -= torch.log(1.0 - squashed_next_actions.pow(2) + 1e-6)
                    next_log_prob = next_log_prob.sum(dim=-1)

                    # build target to keep the probability of taking an action balanced
                    current_alpha = log_alpha.exp().detach()
                    q_target_critic = rewards + gamma * (1 - is_failures) * (q_next - current_alpha * next_log_prob)


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
                    mean, std = model_actor(states)
                    dist = torch.distributions.Normal(mean, std)
                    # rsample supports backward prop for training
                    raw_predicted_actions = dist.rsample()
                    predicted_actions = torch.tanh(raw_predicted_actions)
                    log_prob = dist.log_prob(raw_predicted_actions)
                    log_prob -= torch.log(1.0 - predicted_actions.pow(2) + 1e-6)
                    log_prob = log_prob.sum(dim=-1)

                    # now we get the q values
                    actor_q_1 = model_critic(states, predicted_actions).squeeze(-1)
                    actor_q_2 = model_second_critic(states, predicted_actions).squeeze(-1)
                    actor_q_values = torch.min(actor_q_1, actor_q_2)

                    # actor is maximizing the sum of Q for the trajectory + log_prob balancing
                    current_alpha = log_alpha.exp().detach()
                    loss_actor = (current_alpha * log_prob - actor_q_values).mean()
                    actor_optimizer.zero_grad()
                    loss_actor.backward()
                    torch.nn.utils.clip_grad_norm_(model_actor.parameters(), max_norm=10)
                    actor_optimizer.step()

                    # optimize alpha
                    alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()

                    # sync network
                    polyak_update(model_critic, target_critic, tau=0.005)
                    polyak_update(model_second_critic, target_critic_two, tau=0.005)


        eval_env = gym.make(env_id)


        if verbose and i % test_freq == 0:
            print(f"Self-train itr: {i} | Env step: {total_steps} | Alpha: {log_alpha.exp().detach().squeeze(0)}")
            # SAC is unique in that it is off policy but uses a distribution which all policy gradient algorithms uses
            avg_reward_per_episodes = evaluate_policy_continuous_SAC( model_actor,
                                                                            env=eval_env,
                                                                            solved_threshold=solved_threshold,
                                                                            success_threshold=success_threshold,
                                                                            verbose=verbose,
                                                                            n_episodes=20
                                                                            )
            total_rewards.append(avg_reward_per_episodes)

            if avg_reward_per_episodes >= solved_threshold:
                print(f"\nSolved in {i + 1} iterations!")
                torch.save(model_actor.state_dict(), f"solved models/{env_id}_SAC_solved.pth")
                return losses, total_rewards


    return losses, total_rewards



