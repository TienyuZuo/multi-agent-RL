'''
Author: tienyu tienyu.zuo@nuist.edu.cn
Date: 2024-04-18 17:39:03
LastEditors: tienyu tienyu.zuo@nuist.edu.cn
LastEditTime: 2024-04-19 16:07:39
FilePath: \algo\ddpg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

class ReplayBuffer():
    def __init__(self, max_size=1000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size=256):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        state, next_state, action, reward, done = [], [], [], [], []
        
        for i in ind:
            s, s_prime, a, r, d = self.storage[i]
            state.append(np.array(s, copy=False))
            next_state.append(np.array(s_prime, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            done.append(np.array(d, copy=False))

        return np.array(state), np.array(next_state), np.array(action), np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1)
        
class Agent():
    def __init__(self) -> None:
        self.gamma = 0.99

        self.actor = Actor()
        self.critic = Critic()
        self.actor_target = self.actor
        self.critic_target=  self.critic

        self.buffer = ReplayBuffer()

        self.critic_loss_func = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
    
    def selection_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.actor(state).cpu().data.numpy()
        return action

    def update(self,):
        for update_time in range(100):
            state, next_state, action, reward, done = self.buffer.sample()
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)
        
            actor_loss = -self.critic(state, self.actor(state)).mean()
            critic_loss = self.critic_loss_func(reward + self.gamma * (1-done) * self.critic_target(next_state, self.actor_target(next_state).detach()).detach(), 
                                                self.critic(state, action))
            loss = actor_loss + critic_loss
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            self.soft_target_updata(self.actor_target, self.actor)
            self.soft_target_updata(self.critic_target, self.critic)

    def soft_target_updata(self, target_net, updated_net, tau=0.95):
        for updated_net_param, target_net_param in zip(updated_net.parameters(), target_net.parameters()):
            target_net_param.data.copy_(tau * updated_net_param.data + (1.0-tau) * target_net_param.data)

class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), 
                                 nn.Linear(128, action_dim), nn.Tanh())
    
    def forward(self, state):
        return max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, 128), nn.ReLU(), 
                                 nn.Linear(128, 1))
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))

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

env = Env("Pendulum-v1")
state_dim = env.env.observation_space.shape[0]
action_dim = env.env.action_space.shape[0]
max_action = float(env.env.action_space.high[0])

if __name__ == "__main__":
    agent = Agent()
    
    for epi in range(1000):
        state = env.reset()
        total_reward = 0
        for time in range (10000):
            action = agent.selection_action(state)
            action = (action + np.random.normal(0, 0.1, size=action_dim)).clip(
                        env.env.action_space.low, env.env.action_space.high)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.push((state, next_state, action, reward, done))
            # env.render()
            state = next_state
            
            total_reward += reward

            if done or time >= 9999:
                agent.update()
                if epi % 10 == 0:
                    print("Episode: {} Step is {} Total reward is {}".format(epi, time, total_reward))
                break