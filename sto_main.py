import numpy as np
import random
import time
from collections import defaultdict
from sto_environment import GridWorld
from agent import QLearningAgent
from sto1_export_to_prism import export_to_prism
import pandas as pd

ACTION_SYMBOLS = {
    0: "⬆️",
    1: "➡️",
    2: "⬇️",
    3: "⬅️"
}

def evaluate_policy(env, policy, num_episodes=100):
    success = 0
    danger = 0
    total_reward = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy.get(state, random.choice(env.action_space))
            next_state, reward, done = env.step(action)
            total_reward += reward
            if next_state == env.goal:
                success += 1
            if next_state in env.danger:
                danger += 1
            state = next_state

    return success / num_episodes, danger / num_episodes, total_reward / num_episodes

    print("\n📊 Learned Q-table:")
    for y in range(size):
        for x in range(size):
            state = (x, y)
            q_values = agent.q_table[state]
            q_str = ", ".join(f"{q:.2f}" for q in q_values)
            print(f"State {state}: Q-values = [{q_str}]")
def main():
    # ---- total runtime start ----
    t0_total = time.perf_counter()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to: {seed}")

    size = 20
    start = (0, 0)
    goal = (19, 19)
    danger = [(i, j) for i in range(6, 10) for j in range(6, 10)]  # 中央设置 danger 方块

    env = GridWorld(size=size, start=start, goal=goal, danger=danger)
    agent = QLearningAgent(env)

    episodes = 5000
    rewards = []
    success_count = 0
    danger_count = 0
    converge_episode = None

    # ---- training runtime start ----
    t0_train = time.perf_counter()

    prev_policy = None
    for ep in range(episodes):
        total_reward, visited = agent.learn_one_episode()
        rewards.append(total_reward)

        if goal in visited:
            success_count += 1
        if any(d in visited for d in danger):
            danger_count += 1

        current_policy = agent.extract_policy()
        if prev_policy is not None and current_policy == prev_policy and converge_episode is None:
            converge_episode = ep  # First time policy stabilizes
        prev_policy = current_policy

    # ---- training runtime end ----
    t1_train = time.perf_counter()

    # Fallback if never converged
    if converge_episode is None:
        converge_episode = episodes

    avg_reward = np.mean(rewards)
    success_rate = success_count / episodes
    danger_rate = danger_count / episodes

    print("\n✅ Training Summary:")
    print(f"- Avg Converge Ep   : {converge_episode}")
    print(f"- Avg Final Reward  : {avg_reward:.2f}")
    print(f"- Success Rate      : {success_rate:.2%}")
    print(f"- Danger Rate       : {danger_rate:.2%}")
    print(f"- Training Time     : {t1_train - t0_train:.3f} s")  # 新增：训练耗时

    # 输出策略
    policy = agent.extract_policy()
    print("\n📋 Learned Policy Grid:")
    for y in range(size):
        row = []
        for x in range(size):
            state = (x, y)
            if state == goal:
                row.append("✅")
            elif state in danger:
                row.append("☠️")
            else:
                action = policy[state]
                row.append(ACTION_SYMBOLS[action])
        print(" ".join(row))

    # 导出 PRISM 模型
    export_to_prism(
        policy=policy,
        width=size,
        height=size,
        start=start,
        goal=goal,
        danger_list=danger,
        filename="policy_model.prism"
    )
    print("\n📦 PRISM model exported as policy_model.prism")

    # ---- total runtime end ----
    t1_total = time.perf_counter()
    print(f"- Total Runtime     : {t1_total - t0_total:.3f} s")  # 新增：总耗时

if __name__ == "__main__":
    main()
