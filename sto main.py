from sto_environment import GridWorld
from sto_agent import QLearningAgent
from sto1_export_to_prism import export_to_prism
import random
import numpy as np

ACTION_SYMBOLS = {
    0: "⬆️",
    1: "➡️",
    2: "⬇️",
    3: "⬅️"
}

def main():
    seed = 41
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to: {seed}")

    with open("used_seed.txt", "w") as f:
        f.write(str(seed))

    size = 7
    start = (0, 0)
    goal = (3, 2)
    danger = [ (1, 3), (1, 2)]

    env = GridWorld(size=size, start=start, goal=goal, danger=danger, stochastic=True, noise_prob=0.36)
    agent = QLearningAgent(env)
    agent.learn(episodes=5000)

    print("\n📋 Learned Policy Grid:")
    policy = agent.extract_policy()
    for y in range(size):
        row = []
        for x in range(size):
            state = (x, y)
            if state == goal:
                row.append("✅️")
            elif state in danger:
                row.append("☠️")
            else:
                action = policy[state]
                row.append(ACTION_SYMBOLS[action])
        print(" ".join(row))

    print("\n📊 Learned Q-table:")
    for y in range(size):
        for x in range(size):
            state = (x, y)
            q_values = agent.q_table[state]
            q_str = ", ".join(f"{q:.2f}" for q in q_values)
            print(f"State {state}: Q-values = [{q_str}]")

    # ✅ 在 main() 函数内部导出策略
    export_to_prism(
        policy=policy,
        width=size,
        height=size,
        start=start,
        goal=goal,
        danger_list=danger,
        noise_prob=env.noise_prob,  # 👈 这里不会报错
        filename="policy_model.prism"
    )
    print("\n PRISM model has been successfully exported as policy_model.prism")

if __name__ == "__main__":
    main()
