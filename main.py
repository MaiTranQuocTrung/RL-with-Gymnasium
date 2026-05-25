import gymnasium as gym
import torch
from neural_nets import Network, DuelingNet, REINFORCE_Network
from models import NFQ,DQN, REINFORCE
import numpy as np
from helper_functions import get_all_q

env_id = "CartPole-v1"
env = gym.make(id=env_id, render_mode="human")

state_space = env.observation_space.shape[0]
action_space = env.action_space.n


#define our models
mlp = Network(input_size=state_space, action_space=action_space, hidden_sizes=[256, 256])
dueling = DuelingNet(input_size=state_space, action_space=action_space, trunk_sizes=[512, 256], value_sizes=[128], advantage_sizes=[128])
reinforce_net = REINFORCE_Network(input_size=state_space, action_space=action_space, hidden_sizes=[256, 256], discrete=True)

# train our network
#NFQ.train_model(model=dueling, model_id="mlp", solved_threshold=500, success_threshold=200, env_id=env_id)
#DQN.train_model(model=mlp, model_id="mlp", solved_threshold=500, success_threshold=200, env_id=env_id, eps_steps_decay= 2000)
'''DQN.train_model(model=mlp, model_id="mlp", solved_threshold=260, success_threshold=200,
                env_id=env_id, sync_freq=500, min_sample=2500, model_learning_rate=0.0002,
                self_train_itr=500_000, eps_steps_decay= 150_000)'''
#REINFORCE.train_model(model=reinforce_net, model_id="reinforce_net", solved_threshold=500, success_threshold=200,env_id=env_id)

# run a demo
truncated = terminated = False
state, _ = env.reset()

# load model
state_dict = torch.load('solved models/reinforce_net_CartPole-v1_REINFORCE_solved.pth', weights_only=True)
reinforce_net.load_state_dict(state_dict)
reinforce_net.eval()

while not(truncated or terminated):
    action = np.argmax(get_all_q(state, reinforce_net))
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state
    env.render()
env.close()