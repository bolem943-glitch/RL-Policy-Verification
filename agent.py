import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(env.action_dim))
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, state):
        # 总是选择当前Q值最大的动作
        return np.argmax(self.q_table[state])

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

    def learn(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                best_next_q = np.max(self.q_table[next_state])
                td_target = reward + self.gamma * best_next_q
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_error
                state = next_state
                total_reward += reward
                steps += 1
            if episode % 100 == 0:
                print(f"[Deterministic] Episode {episode}: Reward={total_reward:.1f}, Steps={steps}")

    def extract_policy(self):
        policy = {}
        for x in range(self.env.width):
            for y in range(self.env.height):
                state = (x, y)
                policy[state] = np.argmax(self.q_table[state])
        return policy

