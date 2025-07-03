import os
import gym
import torch
import numpy as np
from collections import defaultdict

# 假设 utils.py 和 plot.py 已正确定义
from utils import CliffWalkingWapper, save_results, make_dir
from plot import plot_rewards

# 确保路径正确
curr_path = os.getcwd()

class SarsaConfig:
    def __init__(self):
        self.seed = 42
        self.algo = 'SARSA'
        self.env = 'CliffWalking-v0'
        self.result_path = curr_path + "/outputs/" + self.env + '/results/'
        self.model_path = curr_path + "/outputs/" + self.env + '/models/'
        self.train_eps = 500  # 增加训练轮数
        self.eval_eps = 300
        self.gamma = 0.9
        self.lr = 0.1  # 提高学习率
        self.epsilon = 0.7  # 提高初始探索率
        self.render_frqc = 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sarsa:
    def __init__(self, state_dim, action_dim, cfg):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon
        self.Q = defaultdict(lambda: np.zeros(action_dim))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, next_action, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state][next_action]
        self.Q[state][action] += self.lr * (target - self.Q[state][action])

    def predict(self, state):
        return np.argmax(self.Q[state])

    def save(self, path):
        import pickle
        make_dir(path, '')
        with open(path + 'sarsa_model.pkl', 'wb') as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path):
        import pickle
        with open(path + 'sarsa_model.pkl', 'rb') as f:
            loaded_q = pickle.load(f)
            self.Q = defaultdict(lambda: np.zeros(self.action_dim), loaded_q)

def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env)
    env = CliffWalkingWapper(env)  # 可临时注释以测试原始环境: env = gym.make(cfg.env)
    env.seed(seed)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = Sarsa(state_dim, action_dim, cfg)
    return env, agent

def train(cfg, env, agent):
    print('Start to train!')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    rewards = []
    running_rewards = []

    for i_ep in range(cfg.train_eps):
        ep_reward = 0
        state = env.reset()
        action = agent.choose_action(state)
        agent.epsilon = max(0.1, cfg.epsilon / (1 + i_ep / 500))  # 指数衰减

        while True:
            next_state, reward, done, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            if i_ep % cfg.render_frqc == 0 and i_ep != 0:
                env.render()
            agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            ep_reward += reward
            if done:
                break

        rewards.append(ep_reward)
        running_rewards.append(0.9 * running_rewards[-1] + 0.1 * ep_reward if running_rewards else ep_reward)
        print(f"Episode:{i_ep+1}/{cfg.train_eps}, reward:{ep_reward:.1f}, epsilon:{agent.epsilon:.3f}")

    print('Complete training!')
    return rewards, running_rewards

def eval(cfg, env, agent):
    print('Start to eval!')
    print(f'Env:{cfg.env}, Algorithm:{cfg.algo}, Device:{cfg.device}')
    agent.load(cfg.model_path)
    rewards = []
    running_rewards = []
    paths = []  # 记录路径

    for i_ep in range(cfg.eval_eps):
        ep_reward = 0
        state = env.reset()
        path = [state]  # 记录当前Episode的路径

        while True:
            action = agent.predict(state)
            next_state, reward, done, _ = env.step(action)
            if i_ep % 3 == 0:
                env.render()
            state = next_state
            ep_reward += reward
            path.append(state)
            if done:
                break

        rewards.append(ep_reward)
        running_rewards.append(0.9 * running_rewards[-1] + 0.1 * ep_reward if running_rewards else ep_reward)
        paths.append(path)
        print(f"Episode:{i_ep+1}/{cfg.eval_eps}, reward:{ep_reward:.1f}")

    # 打印第一个和最后一个Episode的路径
    print('Complete evaling!')
    return rewards, running_rewards

if __name__ == "__main__":
    try:
        import pygame
    except ImportError:
        print("Installing pygame...")
        os.system("pip install pygame")
        import pygame

    cfg = SarsaConfig()
    env, agent = env_agent_config(cfg, cfg.seed)

    # 训练
    rewards, running_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, running_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, running_rewards, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)

    # 测试
    rewards, running_rewards = eval(cfg, env, agent)
    save_results(rewards, running_rewards, tag='eval', path=cfg.result_path)
    plot_rewards(rewards, running_rewards, tag="eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)