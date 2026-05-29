import gymnasium as gym
import torch
from neural_nets import Network, REINFORCE_Network
from models import A3C, A2C
from helper_functions import evaluate_policy_continuous
import torch.multiprocessing as mp

env_id = "HalfCheetah-v5"
env = gym.make(id=env_id, render_mode="human")

state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

#define our models
reinforce_net_policy = REINFORCE_Network(input_size=state_space, action_space=action_space, hidden_sizes=[256, 256], discrete=False)
reinforce_net_value = Network(input_size=state_space, action_space=1, hidden_sizes=[256, 256])

if __name__ == '__main__':
    '''
    mp.set_start_method('spawn')
    A3C.train_model(
            global_net_policy=reinforce_net_policy,
            global_net_value=reinforce_net_value,
            model_str='reinforce_net_cont',
            env_id=env_id,
            solved_threshold=9100,
            success_threshold=7000,
            n_steps=30,
            self_train_itr=100_000,
            gamma=0.99,
            n_workers=6,
            test_freq=50,
            model_policy_learning_rate=0.0001,
            model_value_learning_rate=0.0001,
            entropy_coef= 0.01,
    )
    '''

    A2C.train_model(model_value=reinforce_net_value,
                    model_policy=reinforce_net_policy,
                    model_id="Reinforce_Net",
                    env_id=env_id,
                    solved_threshold=4100,
                    success_threshold=2800,
                    n_steps=40,
                    self_train_itr=100_000,
                    gamma=0.99,
                    n_workers=10,
                    test_freq=50,
                    entropy_coef=0.01,
                    model_policy_learning_rate = 0.0001,
                    model_value_learning_rate = 0.0001,
                    verbose= True,
    )


    # load model
'''
    state_dict = torch.load('solved models/reinforce_net_cont_InvertedPendulum-v5_A3C_solved.pth', weights_only=True)
    reinforce_net_policy.load_state_dict(state_dict)
    reinforce_net_policy.eval()
    
    evaluate_policy_continuous(reinforce_net_policy, solved_threshold=9100.0, success_threshold=8000.0, n_episodes=1,render=True, env_id=env_id)
'''