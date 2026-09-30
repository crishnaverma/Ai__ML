import flappy_bird_gymnasium
import gymnasium as gym
import random
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Agent:

    def __init__(self, param_set):
        self.param = param_set
        with open("parameters.yaml","r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]
        self.mini_batch_size = params["mini_batch_size"]     

        self.loss_fn = nn.MSELoss()
        self.optimizer = None     


    def run(self, is_training = True, render = False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0] #input dimension
        num_actions = env.action_space.n #output dimension
        
        policy_dqn = DQN(num_states, num_actions).to(device)


        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            

        for episode in itertools.count():    
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0

            while not terminated: 
                if is_training and random.random()<epsilon:
                    action = env.action_space.sample()#explore
                    action = torch.tensor(action, dtype=torch.float, device=device)
                else:
                    with torch.no_grad:
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                action = env.action_space.sample()

                # Processing: terminated => done
                next_state, reward, terminated, _, _ = env.step(action.item())

                reward = torch.tensor(reward, dtype=torch.float, device=device)
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))

                episode_reward += reward

            print(f"for episode = {episode+1} with total_reward = {episode_reward}")

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

            #env.close()
