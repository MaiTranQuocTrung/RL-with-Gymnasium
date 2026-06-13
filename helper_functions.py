import torch
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, TransformObservation, TransformReward
import numpy as np
from gymnasium.wrappers import RescaleAction, TransformObservation
from gymnasium.wrappers.vector import NormalizeObservation as VecNormalizeObservation
from gymnasium.wrappers.vector import TransformObservation as VecTransformObservation
from gymnasium.wrappers.vector import NormalizeReward as VecNormalizeReward
from gymnasium.wrappers.vector import TransformReward as VecTransformReward
from gymnasium.wrappers.vector import RescaleAction as VecRescaleAction
'''
    Returns Q(s, a) for all actions given next_state.
    This is used to compute max Q(s', a') in the Bellman equation.
'''


def get_all_q(next_state, model):
    model.eval()
    with torch.no_grad():
        state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)
        return q_values.squeeze(0).numpy()


'''
    This function takes in our model, aka our policy, and tests how well it does given a similar environment.
'''


def evaluate_policy_discrete(
        model,
        solved_threshold,
        success_threshold,
        env_id : str,
        verbose: bool = False,
        n_episodes=100,
):
    env = gym.make(env_id)

    total = 0
    successes = 0

    solved = 0

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                # remember our network outputs ([float32, float32, float32, float32]) but we want (1,6), this is why unsqueeze, this is because pytorch models always take in "batch style data"
                x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                # feed to the model, which returns the best action it thinks
                action = model(x).argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward

        total += episode_reward

        if episode_reward >= success_threshold:
            successes += 1
        if episode_reward >= solved_threshold:
            solved += 1

    env.close()

    avg = total / n_episodes

    if verbose:
        print(f"Avg reward: {avg:.1f}, Successful threshold: {successes}/{n_episodes}, Fully solved: {solved}/{n_episodes}")
    return avg


def evaluate_policy_continous_off_policy(
        model,
        env,
        solved_threshold,
        success_threshold,
        verbose: bool = False,
        n_episodes=20,
):

    total = 0
    successes = 0

    solved = 0

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():

                # this is  single scalar value for an action
                action = action = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward

        total += episode_reward

        if episode_reward >= success_threshold:
            successes += 1
        if episode_reward >= solved_threshold:
            solved += 1

    env.close()

    avg = total / n_episodes

    if verbose:
        print(f"Avg reward: {avg:.1f}, Successful threshold: {successes}/{n_episodes}, Fully solved: {solved}/{n_episodes}")
    return avg


def evaluate_policy_continuous_on_policy(
        model,
        solved_threshold,
        success_threshold,
        env,
        verbose: bool = False,
        n_episodes=100,
        render: bool = False,
):
    total = 0
    successes = 0

    solved = 0

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                x = torch.tensor(state, dtype=torch.float32)

                # feed to the model which returns mean and std
                mean, _ = model(x)

            state, reward, terminated, truncated, _ = env.step(mean.detach().numpy())

            if render:
                env.render()

            done = terminated or truncated

            episode_reward += reward

        total += episode_reward

        if episode_reward >= success_threshold:
            successes += 1
        if episode_reward >= solved_threshold:
            solved += 1

    env.close()

    avg = total / n_episodes

    if verbose:
        print(f"Avg reward: {avg:.1f}, Successful threshold: {successes}/{n_episodes}, Fully solved: {solved}/{n_episodes}")
    return avg


def evaluate_policy_continuous_SAC(
        model,
        solved_threshold,
        success_threshold,
        env,
        verbose: bool = False,
        n_episodes=100,
        render: bool = False,
):
    total = 0
    successes = 0

    solved = 0

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                x = torch.tensor(state, dtype=torch.float32)

                # feed to the model which returns mean and std
                mean, _ = model(x)
                deterministic_action = torch.tanh(mean).detach().numpy()

            state, reward, terminated, truncated, _ = env.step(deterministic_action)

            if render:
                env.render()

            done = terminated or truncated

            episode_reward += reward

        total += episode_reward

        if episode_reward >= success_threshold:
            successes += 1
        if episode_reward >= solved_threshold:
            solved += 1

    env.close()

    avg = total / n_episodes

    if verbose:
        print(f"Avg reward: {avg:.1f}, Successful threshold: {successes}/{n_episodes}, Fully solved: {solved}/{n_episodes}")
    return avg

def polyak_update(main_model, target_model, tau=0.005):
    """
    Softly update target_model parameters toward main_model.
    """
    with torch.no_grad():
        for p_main, p_targ in zip(main_model.parameters(), target_model.parameters()):
            p_targ.data.mul_(1 - tau)
            p_targ.data.add_(tau * p_main.data)

# normalize env states and rewards
def make_env(env_id, animate: bool = False):
    def thunk():
        if animate:
            env = gym.make(env_id, render_mode="human")
        else:
            env = gym.make(env_id)

        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        #env = NormalizeObservation(env)
        #env = TransformObservation(env, lambda obs: np.clip(obs, -10.0, 10.0), env.observation_space)
        return env

    return thunk


def setup_envs(env_id, n_workers, gamma):
    envs = gym.vector.SyncVectorEnv([make_env(env_id) for _ in range(n_workers)])
    envs = VecRescaleAction(envs, min_action=-1.0, max_action=1.0)
    envs = VecNormalizeObservation(envs)
    envs = VecTransformObservation(envs, lambda obs: np.clip(obs, -10.0, 10.0))
    envs = VecNormalizeReward(envs, gamma=gamma)
    envs = VecTransformReward(envs, lambda reward: np.clip(reward, -10.0, 10.0))
    return envs

# run test env
def run_test_env_continous(
        env,
        env_id,
        model,
        n_episodes: int = 1,
        distribution_folder_name : str = "solved dist stats",
        model_folder_name : str = "solved models",
        model_type : str = "PPO",
        on_policy: bool = True,
):

    if on_policy and model_type != "SAC":
        # Apply all the normalization and distribution on the env
        env = NormalizeObservation(env)
        env = TransformObservation(env, lambda obs: np.clip(obs, -10.0, 10.0), env.observation_space)
        unwrapped = env
        while not hasattr(unwrapped, 'obs_rms') and hasattr(unwrapped, 'env'):
            unwrapped = unwrapped.env

        if hasattr(unwrapped, 'obs_rms'):
            unwrapped.obs_rms.mean = np.load(f"{distribution_folder_name}/{model_type}_{env_id}_mean.npy")
            unwrapped.obs_rms.var = np.load(f"{distribution_folder_name}/{model_type}_{env_id}_var.npy")
            unwrapped.obs_rms.count = 1e8
        else:
            print("Warning: Could not find obs_rms to load normalization stats.")

    state_dict = torch.load(f"{model_folder_name}/{env_id}_{model_type}_solved.pth", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    if on_policy or model_type == 'SAC':
        evaluate_policy_continuous_on_policy(model=model, solved_threshold=float('inf'), success_threshold=float('inf'),
                                         env=env, n_episodes=n_episodes, render=True)
    else:
        evaluate_policy_continous_off_policy(model=model, solved_threshold=float('inf'), success_threshold=float('inf'),
                                             n_episodes=n_episodes, env=env)



