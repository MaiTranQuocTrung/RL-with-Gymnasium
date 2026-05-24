import torch
import gymnasium as gym
import numpy as np

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


def evaluate_policy(
        model,
        solved_threshold,
        success_threshold,
        env_id : str,
        verbose: bool = False,
        n_episodes=100,
        record: bool = False
):
    env = gym.make(env_id)

    total = 0
    successes = 0

    solved = 0

    # this is used if you want to record some training footage
    recorded_episode = None

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
        print(f"Avg reward: {avg:.1f}, 200 steps: {successes}/{n_episodes}, Fully solved: {solved}/{n_episodes}")

    if record:
        return avg, recorded_episode

    return avg
