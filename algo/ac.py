'''
Author: tienyu tienyu.zuo@nuist.edu.cn
Date: 2024-04-16 20:49:05
LastEditors: tienyu tienyu.zuo@nuist.edu.cn
LastEditTime: 2024-04-17 21:03:01
FilePath: \algo\ac.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import numpy
import gym
import torch.nn.functional

class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_space_dim, 128), nn.ReLU(),
                                 nn.Linear(128, action_space_dim))
    
    def forward(self, state):
        return torch.nn.functional.softmax(self.net(state))

class Critic(nn.Module):
    def __init__(self,):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_space_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 1))
    
    def forward(self, state):
        return self.net(state)

class Agent():
    def __init__(self) -> None:
        self.pi_log = []
        self.value_logger = []
        self.reward_logger = []
        
        self.actor, self.critic = Actor(), Critic()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = 0.99
    
    def selection_action(self, state):
        state = torch.from_numpy(state).float()
        action_prob = self.actor(state)
        value = self.critic(state)
        action_prob = torch.distributions.Categorical(action_prob)
        action = action_prob.sample()
        action_log = action_prob.log_prob(action)

        self.pi_log.append(action_log)
        self.value_logger.append(value)

        return action.item()


    def update(self):
        reward = 0
        reward_list = []

        # reward_list用于V(s)的真值
        for r in self.reward_logger[::-1]:
            reward = r + self.gamma * reward
            reward_list.insert(0, reward)
        
        Vs = torch.tensor(reward_list)
        Vs = (Vs - Vs.mean()) / (Vs.std() + 1e-7)

        actor_loss = []
        critic_loss = []
        for log_pi, V_pi, V_s  in zip(self.pi_log, self.value_logger, Vs):
            advantage =  V_s - V_pi.item()
            actor_loss.append(-(log_pi * advantage))
            critic_loss.append(advantage.pow(2))

        actor_loss = torch.stack(actor_loss).mean()
        critic_loss = torch.stack(critic_loss).mean()
        loss = actor_loss + critic_loss
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.clear()

    def clear(self,):
        self.pi_log = []
        self.value_logger = []
        self.reward_logger = []

class Env():
    def __init__(self, env_name) -> None:
        self.env_name = env_name
        self.env = gym.make(self.env_name, render_mode="human")
        self.reset()

    def reset(self,):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self,):
        self.env.render()

env = Env('CartPole-v0')
state_space_dim = env.env.observation_space.shape[0]
action_space_dim = env.env.action_space.n

if __name__ == "__main__":
    
    agent = Agent()

    for episode in range(10000):
        state = env.reset()
        for time in range(10000):
            action = agent.selection_action(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            agent.reward_logger.append(reward)
        
            if done or time >= 9999:
                agent.update()
                if episode % 10 == 0:
                    print("Episode {}, step {}".format(episode, time))
                break
            