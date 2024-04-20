import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import argparse
import multiagent.scenarios as scenarios
from itertools import count 
from multiagent.environment import MultiAgentEnv

def get_args():
    parser = argparse.ArgumentParser("MADDPG")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_tag")
    parser.add_argument("--max_episode", type=int, default=100)
    parser.add_argument("--time-steps", type=int, default=200, help="number of time steps")
    # 一个地图最多env.n个agents，用户可以定义min(env.n,num-adversaries)个敌人，剩下的是好的agent
    parser.add_argument("--adversary_nums", type=int, default=1, help="number of adversaries")
    # Core training parameters
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer_size", type=int, default=int(5e5))
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    return args

class ReplayBuffer():
    def __init__(self, config) -> None:
        self.config = config
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size

        self.buffer = dict()
        for agent_id in range(self.config.agent_nums):
            self.buffer[f'state_{agent_id}'] = np.zeros((self.buffer_size, self.config.state_space_size[agent_id]))
            self.buffer[f'action_{agent_id}'] = np.zeros((self.buffer_size, self.config.action_space_size[agent_id]))
            self.buffer[f'reward_{agent_id}'] = np.zeros((self.buffer_size, 1))
            self.buffer[f'next_state_{agent_id}'] = np.zeros((self.buffer_size, self.config.state_space_size[agent_id]))
        
        self.buffer_ptr = 0

    def stor_exp(self, exp):
        buffer_id = int(self.buffer_ptr)
        for agent_id in range(self.config.agent_nums):
            self.buffer[f'state_{agent_id}'][buffer_id] = exp[0][agent_id]
            self.buffer[f'action_{agent_id}'][buffer_id] = exp[1][agent_id]
            self.buffer[f'reward_{agent_id}'][buffer_id] = exp[2][agent_id]
            self.buffer[f'next_state_{agent_id}'][buffer_id] = exp[3][agent_id]
        self.buffer_ptr = (self.buffer_ptr + 1) % self.buffer_size
        
    def sample(self,):
        batch_sample = dict()
        idces = np.random.randint(0, self.buffer_size, self.batch_size)
        for key in self.buffer.keys():
            batch_sample[key] = self.buffer[key][idces]
        return batch_sample

class Agent():
    def __init__(self, idx, config) -> None:
        self.id = idx
        self.config = config

        self.actor = Actor(idx, config.state_space_size[idx], config.action_space_size[idx], config.high_action)
        self.critic = Critic(idx, sum(config.state_space_size), sum(config.action_space_size), config.high_action)

        self.actor_target = Actor(idx, config.state_space_size[idx], config.action_space_size[idx], config.high_action)
        self.critic_target = Critic(idx, sum(config.state_space_size), sum(config.action_space_size), config.high_action)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        self.critic_loss_func = nn.MSELoss()
        
    def selection_action(self, state):
        if np.random.uniform() < self.config.epsilon:
            action = np.random.uniform(-self.config.high_action, self.config.high_action, self.config.action_space_size[self.id])
        else:
            state = torch.tensor(state, dtype=torch.float)
            action = self.actor(state)
            action = action.cpu().data.numpy()
            noise = config.noise_rate * self.config.high_action * np.random.randn(*action.shape)  # gaussian noise
            action = np.clip(action + noise, -self.config.high_action, self.config.high_action)
        # return action.copy()
        return action

    def learn(self, buffer_sample, other_agent):
        local_reward = torch.tensor(buffer_sample[f'reward_{self.id}'], dtype=torch.float32)
        global_state = []
        global_action = []
        global_next_state = []
        for agent_id in range(self.config.agent_nums):
            global_state.append(torch.tensor(buffer_sample[f'state_{agent_id}'], dtype=torch.float32))
            global_action.append(torch.tensor(buffer_sample[f'action_{agent_id}'], dtype=torch.float32))
            global_next_state.append(torch.tensor(buffer_sample[f'next_state_{agent_id}'], dtype=torch.float32))

        action_next_list = []
        other_id = 0
        for agent_id in range(self.config.agent_nums):
            local_next_state = global_next_state[agent_id]
            if agent_id == self.id:
                action_next_list.append(self.actor_target(local_next_state))
            else:
                action_next_list.append(other_agent[other_id].actor_target(local_next_state))
                other_id += 1

        Q = (local_reward + self.config.gamma * self.critic_target(global_next_state, action_next_list)).detach()
        y = self.critic(global_state, global_action)
        critic_loss = self.critic_loss_func(Q, y)
        local_state = global_state[self.id]
        global_action[self.id] = self.actor(local_state)
        actor_loss = (-self.critic(global_state, global_action)).mean()
        loss = actor_loss + critic_loss
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.soft_target_updata(self.actor_target, self.actor)
        self.soft_target_updata(self.critic_target, self.critic)

    def soft_target_updata(self, target_net, updated_net):
        for updated_net_param, target_net_param in zip(updated_net.parameters(), target_net.parameters()):
            target_net_param.data.copy_(self.config.tau * updated_net_param.data + (1.0-self.config.tau) * target_net_param.data)

class Actor(nn.Module):
    def __init__(self, agent_id, state_dim, action_dim, max_action):
        super().__init__()
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.net = nn.Sequential(nn.Linear(self.state_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, action_dim), nn.Tanh())

    def forward(self, local_state):
        return self.max_action * self.net(local_state)

class Critic(nn.Module):
    def __init__(self, agent_id, state_dim, action_dim, max_action) -> None:
        super().__init__()
        self.agent_id = agent_id
        # sum(state)
        self.state_dim = state_dim
        # sum(action)
        self.action_dim = action_dim
        self.max_action = max_action
        self.net = nn.Sequential(nn.Linear(self.state_dim + self.action_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 1))
    
    def forward(self, global_state:list, global_action:list):
        global_state = torch.cat(global_state, dim=1)
        # action nomorlization
        for i in range(len(global_action)):
            global_action[i] /= self.max_action
        global_action = torch.cat(global_action, dim=1)
        critic_inputs = torch.cat([global_state, global_action], dim=1)
        return self.net(critic_inputs)

class MADDPG():
    def __init__(self, config) -> None:
        self.config = config
        self.agent_group = [Agent(idx, config) for idx in range(config.agent_nums)]

        self.buffer = ReplayBuffer(config)
    
    def update(self, update_times=1):
        for i in range(update_times):
            data = self.buffer.sample()
            for agent_id in range(self.config.agent_nums):
                other_agents = (self.agent_group[:agent_id] + self.agent_group[agent_id+1:]).copy()
                self.agent_group[agent_id].learn(data, other_agents)

def Env(config):
    scenario = scenarios.load(config.scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    config.palyer_nums = env.n  # 包含敌人的所有玩家个数
    config.agent_nums = env.n - config.adversary_nums  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
    config.state_space_size = [env.observation_space[i].shape[0] for i in range(config.agent_nums)]  # 每一维代表该agent的obs维度
    action_space_size = []
    for content in env.action_space:
        action_space_size.append(content.n)
    config.action_space_size = action_space_size[:config.agent_nums]  # 每一维代表该agent的act维度
    config.high_action = 1
    config.low_action = -1
    return env, config

config = get_args()
env, config = Env(config)
if __name__ == "__main__":
    maddpg = MADDPG(config)
    for episode in range(config.max_episode):
        global_state = env.reset()
        for times in count():
            action_set= []
            for agent_id in range(config.agent_nums):
                local_state = global_state[agent_id]
                action = maddpg.agent_group[agent_id].selection_action(local_state)
                action_set.append(action)
            # adversary action (random action)
            for adv_id in range(config.adversary_nums):
                action_set.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            print(action_set)
            next_state, reward, done, _ = env.step(action_set)
            # only store the agents' info without adversaries' info
            maddpg.buffer.stor_exp([global_state[:config.agent_nums], action_set[:config.agent_nums], reward[:config.agent_nums], next_state[:config.agent_nums]])
            env.render()
            # for each agent update
            maddpg.update()

            