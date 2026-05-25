import gymnasium as gym
from neural_nets import Network, DuelingNet
from models import NFQ,DQN
import numpy as np
from helper_functions import *

env_id = "LunarLander-v3"
env = gym.make(id=env_id, render_mode="human")

state_space = env.observation_space.shape[0]
action_space = env.action_space.n


#define our models
mlp = Network(input_size=state_space, action_space=action_space, hidden_sizes=[256, 256])
dueling = DuelingNet(input_size=state_space, action_space=action_space, trunk_sizes=[512, 256], value_sizes=[128], advantage_sizes=[128])


# train our network
#NFQ.train_model(model=dueling, model_id="mlp", solved_threshold=500, success_threshold=200, env_id=env_id)
#DQN.train_model(model=mlp, model_id="mlp", solved_threshold=500, success_threshold=200, env_id=env_id, eps_steps_decay= 2000)
DQN.train_model(model=mlp, model_id="mlp", solved_threshold=260, success_threshold=200,
                env_id=env_id, sync_freq=500, min_sample=2500, model_learning_rate=0.0002,
                self_train_itr=500_000, eps_steps_decay= 150_000)

# run a demo
truncated = terminated = False
state, _ = env.reset()
'''
# load model
state_dict = torch.load('mlp_DQN_solved.pth', weights_only=True)
mlp.load_state_dict(state_dict)
mlp.eval()
'''
while not(truncated or terminated):
    action = np.argmax(get_all_q(state, mlp))
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state
    env.render()
env.close()