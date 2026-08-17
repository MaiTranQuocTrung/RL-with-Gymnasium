# RL-with-Gymnasium

PyTorch implementations of classic reinforcement learning algorithms, trained on [Gymnasium](https://gymnasium.farama.org/) environments.

Discrete-action methods (NFQ, DQN/DDQN, REINFORCE, VPG) work on environments such as CartPole. Continuous-control methods (A3C, A2C, PPO, DDPG, TD3, SAC) target MuJoCo tasks such as `Hopper-v5`. Training is started from `main.py` by building the right networks, then calling `train_model` on the algorithm module. Solved policies are written under `solved models/`.

**Algorithms implemented:**
- NFQ
- DQN
- DDQN
- REINFORCE
- VPG
- A3C
- A2C
- PPO
- DDPG
- TD3
- SAC

**Environments:**
- Discrete Gymnasium envs (e.g. CartPole)
- All MuJoCo envs

---

## Setup

```bash
pip install gymnasium torch numpy
# for MuJoCo continuous-control tasks:
pip install gymnasium[mujoco]
```

Create these folders before training (checkpoints and observation stats are written here):

- `solved models/`
- `solved dist stats/` (on-policy algorithms that normalize observations)

Run from the repo root:

```bash
python main.py
```

`main.py` currently evaluates a saved SAC policy on `Hopper-v5`. Uncomment a `train_model(...)` block in that file to train instead.

---

## Networks (`neural_nets.py`)

Build networks from the environment sizes, then pass them into `train_model`.

```python
import gymnasium as gym
from neural_nets import Network, ContinuousNetwork, CriticNetwork, DuelingNet

env = gym.make("Hopper-v5")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]          # continuous
# n_actions = env.action_space.n                # discrete
```

| Class | Role | Typical algorithms |
| --- | --- | --- |
| `Network` | MLP. For Q-learning, `action_space` is the number of discrete actions. For DDPG/TD3, it is the continuous action dim. Set `output_activation=True` for a tanh actor. | NFQ, DQN, REINFORCE, DDPG, TD3 |
| `DuelingNet` | Dueling Q-network (value + advantage heads). Drop-in Q-net for DQN. | DQN / DDQN |
| `ContinuousNetwork` | Gaussian policy: `forward(x)` returns `(mean, std)`. | VPG (continuous), A2C, A3C, PPO, SAC |
| `CriticNetwork` | Q(s, a) critic. `forward(state, action)` returns a scalar Q-value. | DDPG, TD3, SAC |

```python
# Discrete Q-network
q_net = Network(input_size=state_dim, action_space=n_actions, hidden_sizes=[256, 128])

# Continuous Gaussian policy + state-value net (A2C / A3C / PPO / SAC actor)
policy = ContinuousNetwork(input_size=state_dim, action_space=action_dim, hidden_sizes=[512, 512])
value = Network(input_size=state_dim, action_space=1, hidden_sizes=[512, 512])

# Deterministic actor + Q-critic (DDPG / TD3)
actor = Network(input_size=state_dim, hidden_sizes=[256, 256], action_space=action_dim, output_activation=False)
critic = CriticNetwork(state_dim=state_dim, action_dim=action_dim, hidden_sizes=[256, 256])
critic_two = CriticNetwork(state_dim=state_dim, action_dim=action_dim, hidden_sizes=[256, 256])
```

---

## Training

Each algorithm lives in `models/` and exposes `train_model(...)`. Import the module and call it (same pattern as the commented blocks in `main.py`).

Shared arguments across most trainers:

- `env_id` — Gymnasium env id, e.g. `"CartPole-v1"` or `"Hopper-v5"`
- `solved_threshold` / `success_threshold` — episode returns used during periodic evaluation
- `self_train_itr` — number of outer training iterations (episodes or update cycles, depending on the algorithm)
- `gamma` — discount factor
- `verbose` — print training progress

Solved checkpoints are saved under `solved models/` when evaluation reaches `solved_threshold`.

### NFQ

```python
from models import NFQ
from neural_nets import Network

NFQ.train_model(
    model=q_net,
    model_id="cartpole",
    env_id="CartPole-v1",
    solved_threshold=475,
    success_threshold=400,
    self_train_itr=40,
    dataset_size=2000,
    batch_size=128,
    K=40,
    gamma=0.99,
    epsilon=0.5,
    model_learning_rate=0.0005,
    verbose=True,
)
```

Saves to `solved models/{model_id}_NFQ_solved.pth`.

### DQN / DDQN

DDQN is DQN with `double_dqn=True` (the default).

```python
from models import DQN

DQN.train_model(
    model=q_net,
    model_id="cartpole",
    env_id="CartPole-v1",
    solved_threshold=475,
    success_threshold=400,
    eps_steps_decay=10_000,   # required: epsilon decay length in env steps
    double_dqn=True,          # False = vanilla DQN
    self_train_itr=200,
    sync_freq=15,
    buffer_size=50_000,
    batch_size=64,
    gamma=0.99,
    verbose=True,
)
```

Saves to `solved models/{model_id}_{env_id}_DQN_solved.pth`.

### REINFORCE

Discrete policy gradient. Pass a `Network` whose output size is the number of discrete actions.

```python
from models import REINFORCE

REINFORCE.train_model(
    model_id="cartpole",
    env_id="CartPole-v1",
    solved_threshold=475,
    success_threshold=400,
    model=policy_discrete,
    self_train_itr=100_000,
    model_learning_rate=0.0005,
    gamma=0.99,
    verbose=True,
)
```

Saves to `solved models/{model_id}_{env_id}_REINFORCE_solved.pth`.

### VPG

Vanilla policy gradient with a learned value baseline. Discrete (`is_discrete=True`) or continuous (`is_discrete=False`).

```python
from models import VPG

# discrete
VPG.train_model(
    model_id="cartpole",
    env_id="CartPole-v1",
    solved_threshold=475,
    success_threshold=400,
    model_policy_discrete=policy_discrete,
    model_value=value,
    is_discrete=True,
)

# continuous
VPG.train_model(
    model_id="hopper",
    env_id="Hopper-v5",
    solved_threshold=3000,
    success_threshold=2500,
    model_policy=policy,
    model_value=value,
    is_discrete=False,
)
```

Saves to `solved models/{model_id}_{env_id}_VPG_solved.pth`.

### A3C

Asynchronous advantage actor-critic. Uses `torch.multiprocessing`; on Windows call `mp.set_start_method("spawn")` first. Policy and value nets must be `ContinuousNetwork` and `Network`.

```python
import torch.multiprocessing as mp
from models import A3C

if __name__ == "__main__":
    mp.set_start_method("spawn")
    A3C.train_model(
        global_net_policy=policy,
        global_net_value=value,
        model_str="reinforce_net_cont",
        env_id="Hopper-v5",
        solved_threshold=1000,
        success_threshold=900,
        n_steps=30,
        self_train_itr=1_000_000,
        gamma=0.99,
        n_workers=6,
        test_freq=10,
        model_policy_learning_rate=0.0001,
        model_value_learning_rate=0.0001,
        entropy_coef=0.01,
    )
```

Saves to `solved models/{env_id}_A3C_solved.pth`.

### A2C

Synchronous vectorized actor-critic (continuous actions).

```python
from models import A2C

A2C.train_model(
    model_value=value,
    model_policy=policy,
    env_id="Hopper-v5",
    solved_threshold=1000,
    success_threshold=900,
    n_steps=30,
    self_train_itr=10_000,
    gamma=0.99,
    n_workers=10,
    test_freq=10,
    entropy_coef=0.01,
    model_policy_learning_rate=0.0003,
    model_value_learning_rate=0.0003,
    lamda=0.95,
    verbose=True,
)
```

Saves to `solved models/{env_id}_A2C_solved.pth`.

### PPO

Clipped PPO on vectorized continuous environments.

```python
from models import PPO

PPO.train_model(
    model_value=value,
    model_policy=policy,
    env_id="Hopper-v5",
    solved_threshold=-200,
    success_threshold=-300,
    n_steps=2048,
    self_train_itr=1000,
    gamma=0.999,
    n_workers=10,
    test_freq=1,
    entropy_coef=0.01,
    model_policy_learning_rate=0.0003,
    model_value_learning_rate=0.0003,
    mini_batch_size=256,
    n_epochs=10,
    clip_coef=0.2,
    verbose=True,
)
```

Saves to `solved models/{env_id}_PPO_solved.pth`.

### DDPG

Deterministic actor-critic. Actor is `Network`; critic is `CriticNetwork`.

```python
from models import DDPG

DDPG.train_model(
    model_actor=actor,
    model_critic=critic,
    env_id="Hopper-v5",
    solved_threshold=-200,
    success_threshold=-350,
    batch_size=128,
    self_train_itr=1000,
    sigma=0.2,
    test_freq=1,
    gamma=0.99,
)
```

Saves to `solved models/{env_id}_DDPG_solved.pth`.

### TD3

Twin delayed DDPG. Pass two critics.

```python
from models import TD3

TD3.train_model(
    model_actor=actor,
    model_critic=critic,
    model_second_critic=critic_two,
    env_id="Hopper-v5",
    solved_threshold=4300,
    success_threshold=3800,
    batch_size=256,
    self_train_itr=5000,
    sigma=0.2,
    test_freq=20,
    actor_sync_freq=2,
)
```

Saves to `solved models/{env_id}_TD3_solved.pth`.

### SAC

Soft actor-critic. Actor is `ContinuousNetwork`; two `CriticNetwork`s.

```python
from models import SAC

SAC.train_model(
    model_actor=policy,
    model_critic=critic,
    model_second_critic=critic_two,
    env_id="Hopper-v5",
    solved_threshold=3500,
    success_threshold=2800,
    batch_size=256,
    self_train_itr=5000,
    test_freq=20,
)
```

Saves to `solved models/{env_id}_SAC_solved.pth`.

---

## Evaluation and helpers (`helper_functions.py`)

### Watch a trained continuous policy

`run_test_env_continous` loads `{model_folder_name}/{env_id}_{model_type}_solved.pth`, optionally restores observation normalization from `solved dist stats/`, and renders episodes.

```python
from helper_functions import run_test_env_continous

env = gym.make(env_id, render_mode="human")
run_test_env_continous(
    env=env,
    env_id=env_id,
    model=policy,                 # same architecture as the saved checkpoint
    model_type="SAC",             # "PPO", "A2C", "A3C", "DDPG", "TD3", "SAC", ...
    n_episodes=1,
    on_policy=True,               # False for DDPG / TD3 (deterministic actor)
    distribution_folder_name="solved dist stats",
    model_folder_name="solved models",
)
```

Use `on_policy=True` for A2C / A3C / PPO / SAC (Gaussian policies). Use `on_policy=False` for DDPG / TD3.

### Score a policy without rendering

```python
from helper_functions import (
    evaluate_policy_discrete,
    evaluate_policy_continuous_on_policy,
    evaluate_policy_continous_off_policy,
    evaluate_policy_continuous_SAC,
)

# discrete Q / policy net: greedy argmax
evaluate_policy_discrete(
    model=q_net,
    env_id="CartPole-v1",
    solved_threshold=475,
    success_threshold=400,
    n_episodes=100,
    verbose=True,
)

# Gaussian policy: take the mean action
evaluate_policy_continuous_on_policy(
    model=policy,
    env=env,
    solved_threshold=3000,
    success_threshold=2500,
    n_episodes=20,
    verbose=True,
    render=False,
)

# deterministic actor (DDPG / TD3)
evaluate_policy_continous_off_policy(
    model=actor,
    env=env,
    solved_threshold=3000,
    success_threshold=2500,
    n_episodes=20,
    verbose=True,
)

# SAC: tanh(mean) as the deterministic action
evaluate_policy_continuous_SAC(
    model=policy,
    env=env,
    solved_threshold=3000,
    success_threshold=2500,
    n_episodes=20,
    verbose=True,
)
```

Each of these returns the average episode return.

### Other helpers

```python
from helper_functions import get_all_q, polyak_update, make_env, setup_envs

q_values = get_all_q(next_state, q_net)   # Q(s, a) for every discrete action
polyak_update(main_model, target_model, tau=0.005)

env = make_env("Hopper-v5", animate=False)()   # single env, actions rescaled to [-1, 1]
envs = setup_envs("Hopper-v5", n_workers=8, gamma=0.99)  # vector env with obs/reward norm
```
