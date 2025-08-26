from sto_environment import GridWorld
from sto_agent import QLearningAgent
from sto_export_to_prism import export_to_prism
import random
import numpy as np
import time  # 计时

ACTION_SYMBOLS = {
    0: "⬆️",
    1: "➡️",
    2: "⬇️",
    3: "⬅️"
}

def policies_equal(policy1, policy2, tol=1e-3):
    """比较两个 'state -> prob array' 策略是否近似相等。"""
    if policy1.keys() != policy2.keys():
        return False
    for state in policy1:
        if not np.allclose(policy1[state], policy2[state], atol=tol):
            return False
    return True

def main():
    # ---- total runtime start ----
    t0_total = time.perf_counter()

    # 随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to: {seed}")
    with open("used_seed.txt", "w") as f:
        f.write(str(seed))

    # 环境配置
    size = 20
    start = (0, 0)
    goal = (19, 19)
    danger = [(i, j) for i in range(6, 10) for j in range(6, 10)]  # 中央 danger 区域

    env = GridWorld(size=size, start=start, goal=goal, danger=danger)
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, temperature=0.5)

    # 统计信息
    episodes = 5000
    max_steps = 500
    total_rewards = []
    success_count = 0
    danger_count = 0
    # converge_episode = None
    # prev_policy = None

    # ---- training runtime start ----
    t0_train = time.perf_counter()

    # 训练循环
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        visited_danger = False
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)

            total_reward += reward
            if next_state in danger:
                visited_danger = True

            state = next_state
            steps += 1

        total_rewards.append(total_reward)
        if state == goal:
            success_count += 1
        if visited_danger:
            danger_count += 1

        # 若需要“分布收敛”检测，可解除以下注释
        # current_policy = agent.extract_stochastic_policy()
        # if prev_policy is not None and policies_equal(current_policy, prev_policy) and converge_episode is None:
        #     converge_episode = episode
        # prev_policy = current_policy

    # ---- training runtime end ----
    t1_train = time.perf_counter()

    # 汇总统计
    avg_reward = float(np.mean(total_rewards))
    success_rate = 100.0 * success_count / episodes
    danger_rate = 100.0 * danger_count / episodes
    # converge_ep = converge_episode if converge_episode is not None else "Not Converged"

    print("\n✅ Training Summary:")
    # print(f"- Avg Converge Ep   : {converge_ep}")
    print(f"- Avg Final Reward  : {avg_reward:.2f}")
    print(f"- Success Rate      : {success_rate:.2f}%")
    print(f"- Danger Rate       : {danger_rate:.2f}%")
    print(f"- Training Time     : {t1_train - t0_train:.3f} s")

    # 提取 stochastic policy（state -> 概率分布）
    stoch_policy = agent.extract_stochastic_policy()

    # 用 argmax(probabilities) 渲染人类可读的动作符号
    print("\n📋 Learned Policy Grid (argmax view):")
    for y in range(size):
        row = []
        for x in range(size):
            state = (x, y)
            if state == goal:
                row.append("✅")
            elif state in danger:
                row.append("☠️")
            else:
                probs = stoch_policy.get(state)
                if probs is not None:
                    a = int(np.argmax(probs))
                else:
                    # 若该状态未出现在策略中，则用当前 Q 表的贪婪动作作为回退
                    q_values = agent.q_table.get(state, np.zeros(env.action_dim))
                    a = int(np.argmax(q_values))
                row.append(ACTION_SYMBOLS[a])
        print(" ".join(row))

    # ✅ 导出 PRISM（使用概率分布）
    export_to_prism(
        policy=stoch_policy,
        width=size,
        height=size,
        start=start,
        goal=goal,
        danger_list=danger,
        filename="policy_model.prism"
    )
    print("\n✅ PRISM model exported as policy_model.prism")

    # ---- total runtime end ----
    t1_total = time.perf_counter()
    print(f"- Total Runtime     : {t1_total - t0_total:.3f} s")

if __name__ == "__main__":
    main()
