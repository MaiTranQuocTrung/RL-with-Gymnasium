import copy
import torch
import gymnasium as gym
import torch.multiprocessing as mp
from helper_functions import evaluate_policy_continuous_on_policy, make_env
from gymnasium.wrappers import NormalizeObservation, TransformObservation
import numpy as np

class SharedRMSProp(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-4, alpha=0.99, eps=1e-8):
        super().__init__(params, lr=lr, alpha=alpha, eps=eps)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1).share_memory_()
                state['square_avg'] = torch.zeros_like(p).share_memory_()


def _worker(
        rank: int,
        model_id: str,
        env_id,
        global_optim_value : torch.optim.Optimizer, # we use a shared optimizer
        global_optim_policy : torch.optim.Optimizer,
        solved_threshold: float,
        success_threshold: float,
        solved_event: mp.Event,
        test_freq: int,
        entropy_coef: float,
        n_steps: int = 30,
        model_policy=None,
        model_value=None,
        self_train_itr: int = 100_000,
        gamma: float = 0.99,
        verbose: bool = True,
):
    env = make_env(env_id)()

    # stats tracker
    env_steps = 0

    # local model copies
    local_model_policy = copy.deepcopy(model_policy)
    local_model_value = copy.deepcopy(model_value)

    # Because we're doing n-step, when we finish our n step it is actually most of the time that termination does not happen
    # This means the agent can pickup where it left off at self-train itr time t+1
    state, _ = env.reset()
    terminated = truncated = False

    for i in range(self_train_itr):
        if solved_event.is_set():
            return
        # if the env has been terminated or truncated start a new one
        if terminated or truncated:
            state, _ = env.reset()
            terminated = truncated = False

        local_model_policy.load_state_dict(model_policy.state_dict())
        local_model_value.load_state_dict(model_value.state_dict())

        # n-step style collection
        episode = []
        env_n_steps = 0

        while not (terminated or truncated or env_n_steps >= n_steps):
            with torch.no_grad():
                # get std and mean for Gaussian dist
                mean, std = local_model_policy(torch.tensor(state, dtype=torch.float32))
                # create Gaussian
                dist = torch.distributions.Normal(loc=mean, scale=std)
                # Sample to get action
                action = dist.sample()

            # avoids illegal actions
            clipped_action = action.clamp(env.action_space.low[0], env.action_space.high[0])

            next_state, reward, terminated, truncated, info = env.step(clipped_action.detach().numpy())

            # storing termination to know if to bootstrap
            episode.append((state, action, reward, next_state, float(terminated)))
            env_steps += 1
            env_n_steps += 1

            state = next_state

        # calculate n-steps return
        n_steps_collection = []

        # determine if we need to bootstrap
        with torch.no_grad():
            last_state, _, _, last_next, last_terminated = episode[-1]
            if last_terminated:
                value_state_bootstrap = 0.0
            else:
                value_state_bootstrap = local_model_value(torch.tensor(last_next, dtype=torch.float32)).item() # V(s')

        for (state, action, reward, next_state, terminated) in reversed(episode):
            value_state_bootstrap = reward + gamma * value_state_bootstrap
            n_steps_collection.append(value_state_bootstrap)

        # re-reversed for normal use
        n_steps_collection = list(reversed(n_steps_collection))
        n_steps_collection = torch.tensor(n_steps_collection, dtype=torch.float32)

        total_loss_policy = 0.0
        total_loss_value = 0.0

        # Calculate advantage
        advantages = []
        for (state, _, _, _, _), n_steps_return in zip(episode, n_steps_collection):
            value_pred = local_model_value(torch.tensor(state, dtype=torch.float32)).squeeze() # V(s)
            # n step return - V(s)
            advantages.append(n_steps_return - value_pred.detach())
        advantages = torch.stack(advantages)

        # 2. Normalize the advantages
        '''
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.clamp(advantages, -2.0, 2.0)
        '''

        # Update our model
        # n_step return is actually also our A(s,a)
        for ((state, action, _, _, _), n_steps_return), adv in zip(zip(episode, n_steps_collection), advantages):
            mean, std = local_model_policy(torch.tensor(state, dtype=torch.float32))
            dist = torch.distributions.Normal(loc=mean, scale=std)

            # get entropy
            entropy = dist.entropy().sum()
            weighted_entropy = entropy_coef * entropy

            # V(s)
            value_pred = local_model_value(torch.tensor(state, dtype=torch.float32)).squeeze()

            total_loss_policy += -(adv * dist.log_prob(action).sum() + weighted_entropy)

            target_tensor = torch.tensor(n_steps_return, dtype=torch.float32)
            total_loss_value += torch.nn.functional.mse_loss(value_pred, target_tensor)

        loss_policy = total_loss_policy / len(episode)
        loss_value = total_loss_value / len(episode)

        local_model_policy.zero_grad()
        loss_policy.backward()
        local_model_value.zero_grad()
        loss_value.backward()

        # clip grad
        torch.nn.utils.clip_grad_norm_(local_model_policy.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(local_model_value.parameters(), max_norm=1)

        global_optim_policy.zero_grad()
        global_optim_value.zero_grad()

        for lp, gp in zip(local_model_policy.parameters(), model_policy.parameters()):
            if lp.grad is not None:
                gp.grad = lp.grad.clone()  # Use _grad to inject into optimizer safely

        for lp, gp in zip(local_model_value.parameters(), model_value.parameters()):
            if lp.grad is not None:
                gp.grad = lp.grad.clone()

        global_optim_policy.step()
        global_optim_value.step()

        if i % test_freq == 0 and rank == 0:
            if verbose:
                print(f"Self-train itr: {i} | Env step: {env_steps}")

            eval_env = gym.make(env_id)
            eval_env = NormalizeObservation(eval_env)
            eval_env = TransformObservation(eval_env, lambda obs: np.clip(obs, -10.0, 10.0), eval_env.observation_space)

            train_env = env
            while not hasattr(train_env, 'obs_rms') and hasattr(train_env, 'env'):
                train_env = train_env.env

            eval_env_unwrapped = eval_env
            while not hasattr(eval_env_unwrapped, 'obs_rms') and hasattr(eval_env_unwrapped, 'env'):
                eval_env_unwrapped = eval_env_unwrapped.env

            if hasattr(train_env, 'obs_rms') and hasattr(eval_env_unwrapped, 'obs_rms'):
                eval_env_unwrapped.obs_rms.mean = train_env.obs_rms.mean.copy()
                eval_env_unwrapped.obs_rms.var = train_env.obs_rms.var.copy()
                eval_env_unwrapped.obs_rms.count = train_env.obs_rms.count

            avg_reward_per_ep = evaluate_policy_continuous_on_policy(model_policy, solved_threshold=solved_threshold,
                                                                     success_threshold=success_threshold, env=eval_env,
                                                                     verbose=verbose, n_episodes=20)
            '''
            if avg_reward_per_ep >= success_threshold:
                torch.save(
                    model_policy.state_dict(),
                    f"checkpoint models/{model_id}_{env_id}_rank{rank}_checkpoint_{avg_reward_per_ep:.0f}.pth"
                )
                '''

            if avg_reward_per_ep >= solved_threshold:
                np.save(f"solved dist stats/A3C_{env_id}_mean.npy", eval_env_unwrapped.obs_rms.mean)
                np.save(f"solved dist stats/A3C_{env_id}_var.npy", eval_env_unwrapped.obs_rms.var)
                solved_event.set()
                return



def train_model(
        global_net_policy,
        global_net_value,
        model_str: str,
        env_id: str,
        solved_threshold: float,
        success_threshold: float,
        test_freq: int,
        entropy_coef: float = 0.01,
        n_steps: int = 30,
        self_train_itr: int = 100_000,
        gamma: float = 0.99,
        n_workers: int = 3,
        model_policy_learning_rate: float = 0.0001,
        model_value_learning_rate: float = 0.0001,
):
    global_net_policy.share_memory()
    global_net_value.share_memory()
    global_policy_optimizer = SharedRMSProp(global_net_policy.parameters(), lr=model_policy_learning_rate)
    global_value_optimizer = SharedRMSProp(global_net_value.parameters(), lr=model_value_learning_rate)
    processes = []

    solved_event = mp.Event()

    for rank in range(n_workers):
        p = mp.Process(
            target=_worker,
            args=(
                rank,
                model_str,
                env_id,
                global_value_optimizer,
                global_policy_optimizer,
                solved_threshold,
                success_threshold,
                solved_event,
                test_freq,
            ),
            kwargs={
                'n_steps': n_steps,
                'model_policy': global_net_policy,
                'model_value': global_net_value,
                'self_train_itr': self_train_itr,
                'gamma': gamma,
                'entropy_coef': entropy_coef,
            }
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # all workers done — save the global model
    if solved_event.is_set():
        torch.save(global_net_policy.state_dict(), f"solved models/{env_id}_A3C_solved.pth")
        print("Solved! Model saved.")





