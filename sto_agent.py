import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, temperature=1.0):
        """
        :param temperature: softmax 温度参数，越小越贪婪
        """
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(env.action_dim))
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature

    def softmax(self, q_values):
        """对 Q 值做 softmax，返回动作概率分布"""
        q_values = np.array(q_values)
        q_values -= np.max(q_values)  # 避免数值爆炸
        exp_q = np.exp(q_values / self.temperature)
        probs = exp_q / np.sum(exp_q)
        return probs

    def choose_action(self, state):
        """根据 softmax 分布随机采样动作"""
        probs = self.softmax(self.q_table[state])
        return np.random.choice(self.env.action_space, p=probs)

    def learn_one_episode(self, max_steps=100):
        state = self.env.reset()
        total_reward = 0
        steps = 0
        done = False
        visited_states = set()

        while not done and steps < max_steps:
            visited_states.add(state)
            action = self.choose_action(state)
            next_state, reward, done = self.env.step(action)

            best_next_q = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * best_next_q
            td_error = td_target - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * td_error

            state = next_state
            total_reward += reward
            steps += 1

        visited_states.add(state)
        return total_reward, visited_states

    def learn(self, episodes=5000, max_steps=100):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            steps = 0
            while not done and steps < max_steps:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                best_next_q = np.max(self.q_table[next_state])
                td_target = reward + self.gamma * best_next_q
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_error
                state = next_state
                steps += 1

    def extract_stochastic_policy(self):
        """提取最终 softmax 策略（每个状态对应一个动作概率分布）"""
        policy = {}
        for state, q_values in self.q_table.items():
            policy[state] = self.softmax(q_values)
        return policy

    def update(self, state, action, reward, next_state):
        best_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def extract_greedy_policy(self):
        """提取贪婪策略（每个状态对应最优动作）"""
        policy = {}
        for state, q_values in self.q_table.items():
            best_action = int(np.argmax(q_values))
            policy[state] = best_action
        return policy
