import gymnasium as gym
from neural_nets import Network, ContinuousNetwork, CriticNetwork
from models import A3C, A2C, PPO, DDPG, TD3, SAC
from helper_functions import run_test_env_continous, evaluate_policy_continuous_SAC


env_id = "Hopper-v5"
env = gym.make(env_id, render_mode='human')

state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

#define our models
reinforce_net_policy = ContinuousNetwork(input_size=state_space, action_space=action_space,
                                         hidden_sizes=[512, 512])
reinforce_net_value = Network(input_size=state_space, action_space=1, hidden_sizes=[512, 512])

actor_network = Network(input_size=state_space, hidden_sizes=[256, 256], action_space=action_space, output_activation=False)
critic_network = CriticNetwork(state_dim=state_space, action_dim=action_space, hidden_sizes=[256, 256])
critic_two = CriticNetwork(state_dim=state_space, action_dim=action_space, hidden_sizes=[256, 256])
if __name__ == '__main__':
    '''
    mp.set_start_method('spawn')
    A3C.train_model(
            global_net_policy=reinforce_net_policy,
            global_net_value=reinforce_net_value,
            model_str='reinforce_net_cont',
            env_id=env_id,
            solved_threshold=1000,
            success_threshold=900,
            n_steps=30,
            self_train_itr=1_000_000,
            gamma=0.99,
            n_workers=6,
            test_freq=10,
            model_policy_learning_rate=0.0001,
            model_value_learning_rate=0.0001,
            entropy_coef= 0.01,
    )
    '''
    '''
    A2C.train_model(model_value=reinforce_net_value,
                    model_policy=reinforce_net_policy,
                    env_id=env_id,
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
                    verbose=True,
                    )
    '''
    '''PPO.train_model(model_value=reinforce_net_value,
                    model_policy=reinforce_net_policy,
                    env_id=env_id,
                    solved_threshold=-200,
                    success_threshold=-300,
                    n_steps=2048,
                    self_train_itr=1000,
                    gamma=0.999,
                    n_workers=10,
                    test_freq=1,
                    entropy_coef=0.01,
                    model_policy_learning_rate = 0.0003,
                    model_value_learning_rate = 0.0003,
                    verbose= True,
                    mini_batch_size=256,
    )'''
    '''
    DDPG.train_model(
        model_actor = actor_network,
        model_critic=critic_network,
        env_id=env_id,
        solved_threshold=-200,
        success_threshold=-350,
        batch_size=128,
        self_train_itr=1000
    )
    '''
    '''
    TD3.train_model(
        model_actor=actor_network,
        model_critic=critic_network,
        model_second_critic=critic_two,
        env_id=env_id,
        solved_threshold=4300,
        success_threshold=3800,
        batch_size=256,
        self_train_itr=5000,
        sigma=0.2,
        test_freq=20,
    )
    '''
    '''
    SAC.train_model(
        model_actor=reinforce_net_policy,
        model_critic=critic_network,
        model_second_critic=critic_two,
        env_id=env_id,
        solved_threshold=3500,
        success_threshold=2800,
        batch_size=256,
        self_train_itr=5000,
        test_freq=20,
    )
    '''


    run_test_env_continous(env=env,env_id=env_id,model=reinforce_net_policy,
                           model_type="SAC", n_episodes=1, on_policy=True)
