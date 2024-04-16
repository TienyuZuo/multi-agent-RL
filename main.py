'''
Author: tienyu tienyu.zuo@nuist.edu.cn
Date: 2024-04-16 16:49:36
LastEditors: tienyu tienyu.zuo@nuist.edu.cn
LastEditTime: 2024-04-16 20:34:53
FilePath: \multi-agent-RL\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np
import gym
import torch

import torch.nn as nn
import torch.nn.functional

class Agent():
    def __init__(self) -> None:
        self.policy = Policy()

        self.gamma = 0.99

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        # self.loss PG的目标函数是最大化V(s)，对V求梯度中涉及到对策略求梯度，所以叫policy gradient算法
        self.reward_logger = []
        self.pai_prob = []

    def __clear(self):
        self.reward_logger = []
        self.pai_prob = []

    def selection_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action_prob = self.policy(state)
        action_distribution = torch.distributions.Categorical(action_prob)
        action = action_distribution.sample()
        self.pai_prob.append(action_distribution.log_prob(action))
        
        return action.item()
    
    def update(self):
        # PG算法(REINFORCE算法)的核心是完成一局，获得Q(s,a)，才好对traj中的记录进行更新(梯度上升)
        reward_list = []
        reward = 0
        for r in self.reward_logger[::-1]:
            reward = r + self.gamma * reward 
            reward_list.insert(0, reward)
        reward_list = torch.tensor(reward_list)
        reward_list = (reward_list - reward_list.mean()) / (reward_list.std() + 1e-7)

        loss = []
        for pai_log, reward in zip(self.pai_prob, reward_list):
            loss.append(-(pai_log * reward))
            
        loss = torch.cat(loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.__clear()
    
class Env():
    def __init__(self, env_name) -> None:
        self.env_name = env_name
        self.env = gym.make(self.env_name, render_mode="human")
        self.rest()

    def rest(self,):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self,):
        self.env.render()

class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(4, 128),
                                nn.ReLU(),
                                nn.Linear(128,2))

    def forward(self, state):
        action_prob = self.fc(state)
        return torch.nn.functional.softmax(action_prob, dim=1)
    
        
if __name__ == "__main__":
    max_episode = 1000
    max_quit_times = 1000
    
    agent = Agent()
    env = Env('CartPole-v0')
    for episode in range(max_episode):
        state = env.rest()
        for time in range(max_quit_times):
            action = agent.selection_action(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            # agent.memory_buffer.add_exp(state, action, reward, next_state)
            state = next_state
            agent.reward_logger.append(reward)
            
            if done or time == max_quit_times-1:
                agent.update()
                if episode % 10 == 0:
                    print("episode {}, step is {}".format(episode, time))
                break
