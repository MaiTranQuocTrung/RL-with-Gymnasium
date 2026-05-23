import gymnasium as gym
from neural_nets import Network, DuelingNet
from models import NFQ,DQN

env_id = "CartPole-v1"
env = gym.make(id=env_id, render_mode="human")

state_space = env.observation_space.shape[0]
action_space = env.action_space.n


#define our models
mlp = Network(input_size=state_space, action_space=action_space, hidden_size=512, n_layers=2)
dueling = DuelingNet(input_size=state_space, action_space=action_space)


# train our network
#NFQ.train_model(model=mlp, model_id="mlp", env_id=env_id)
DQN.train_model(model=mlp, model_id="mlp", env_id=env_id)