import os
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from utils import CliffWalkingWapper, save_results, make_dir
from plot import plot_rewards

curr_path = os.path.dirname(__file__)

class DQNConfig:
    """DQN训练相关参数"""
    def __init__(self):
        self.seed = 0
        self.algo = 'DQN'
        self.env = 'CliffWalking-v0'
        self.result_path = curr_path + "/outputs/" + self.env + '/' + '/results/'
        self.model_path = curr_path + "/outputs/" + self.env + '/' + '/models/'
        self.train_eps = 1500
        self.eval_eps = 1000
        self.gamma = 0.9
        self.lr = 0.0005
        self.epsilon_start = 0.5
        self.epsilon_end = 0.05
        self.epsilon_decay = 500
        self.eval_epsilon = 0.05
        self.batch_size = 32
        self.memory_capacity = 2000
        self.target_update = 50
        self.hidden_dim = 64
        self.render_frqc = 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Q网络，双隐藏层MLP"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQN:
    """DQN算法"""
    def __init__(self, state_dim, action_dim, cfg):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.policy_net = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_net = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr, weight_decay=1e-5)
        self.memory = deque(maxlen=cfg.memory_capacity)
        self.step_count = 0

    def choose_action(self, state, episode, is_eval=False):
        epsilon = self.cfg.eval_epsilon if is_eval else \
                  self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
                  np.exp(-1. * episode / self.cfg.epsilon_decay)
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.cfg.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.cfg.batch_size:
            return

        transitions = random.sample(self.memory, self.cfg.batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.cfg.device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.cfg.device)
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.cfg.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.cfg.device)
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.cfg.device)

        current_q = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            target_q = reward_batch + self.cfg.gamma * next_q * (1 - done_batch)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.cfg.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def predict(self, state):
        return self.choose_action(state, episode=0, is_eval=True)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'dqn_model.pth')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'dqn_model.pth'))
        self.target_net.load_state_dict(self.policy_net.state_dict())

def one_hot_encode(state, state_dim):
    vec = np.zeros(state_dim)
    vec[state] = 1
    return vec

def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env)
    env = CliffWalkingWapper(env)  # 启用turtle可视化
    env.seed(seed)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = DQN(state_dim, action_dim, cfg)
    return env, agent

def train(cfg, env, agent):
    print('Start to train!')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    running_rewards = []
    max_steps = 50

    for i_ep in range(cfg.train_eps):
        ep_reward = 0
        state = env.reset()
        state = one_hot_encode(state, env.observation_space.n)
        action = agent.choose_action(state, i_ep)
        step = 0
        path = []

        while True:
            path.append((state.argmax(), action))
            next_state, reward, done, _ = env.step(action)
            next_state = one_hot_encode(next_state, env.observation_space.n)
            next_action = agent.choose_action(next_state, i_ep)

            if i_ep % cfg.render_frqc == 0 and i_ep != 0:
                env.render()

            agent.update(state, action, reward, next_state, done)
            state = next_state
            action = next_action
            ep_reward += reward
            step += 1

            if done or step >= max_steps:
                if step >= max_steps:
                    print(f"Episode {i_ep + 1} terminated due to max steps, path: {path[-10:]}")
                break

        rewards.append(ep_reward)
        running_rewards.append(0.9 * running_rewards[-1] + 0.1 * ep_reward if running_rewards else ep_reward)
        print(f"Episode:{i_ep + 1}/{cfg.train_eps}, reward:{ep_reward:.1f}")

    print('Complete training!')
    env.close()
    return rewards, running_rewards

def eval(cfg, env, agent):
    print('Start to eval!')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    running_rewards = []
    max_steps = 50

    for i_ep in range(cfg.eval_eps):
        ep_reward = 0
        state = env.reset()
        state = one_hot_encode(state, env.observation_space.n)
        step = 0
        path = []

        while True:
            action = agent.predict(state)
            path.append((state.argmax(), action))
            next_state, reward, done, _ = env.step(action)
            next_state = one_hot_encode(next_state, env.observation_space.n)

            if i_ep % 3 == 0:
                env.render()

            state = next_state
            ep_reward += reward
            step += 1

            if done or step >= max_steps:
                if step >= max_steps:
                    print(f"Episode {i_ep + 1} terminated due to max steps, path: {path[-10:]}")
                break

        rewards.append(ep_reward)
        running_rewards.append(0.9 * running_rewards[-1] + 0.1 * ep_reward if running_rewards else ep_reward)
        print(f"Episode:{i_ep + 1}/{cfg.eval_eps}, reward:{ep_reward:.1f}")

    print('Complete evaling!')
    env.close()
    return rewards, running_rewards

if __name__ == "__main__":
    try:
        import turtle
        import pygame
    except ImportError:
        print("Installing turtle and pygame...")
        os.system("pip install turtle pygame")
        import turtle
        import pygame

    cfg = DQNConfig()
    env, agent = env_agent_config(cfg, cfg.seed)

    rewards, running_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, running_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, running_rewards, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)

    rewards, running_rewards = eval(cfg, env, agent)
    save_results(rewards, running_rewards, tag='eval', path=cfg.result_path)
    plot_rewards(rewards, running_rewards, tag="eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)