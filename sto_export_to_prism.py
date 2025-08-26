import numpy as np

def export_to_prism(policy, width, height, start, goal, danger_list,
                    filename="policy_model.prism", noise=0.3, max_steps=200):
    """
    导出 policy 为 PRISM DTMC 模型，带最大步数限制。
    支持：
    - Greedy Policy: dict[(x,y)] = action (int)，主动作 = 1-noise，其余动作平分 noise。
    - Stochastic Policy: dict[(x,y)] = [p_up, p_right, p_down, p_left]
    :param noise: 在 greedy 策略下，主动作的概率为 (1-noise)，其余三动作平分 noise
    :param max_steps: 最大允许步数，超出后进入“终止状态”
    """

    with open(filename, "w") as f:
        f.write("dtmc\n\n")
        f.write("module grid\n")
        f.write(f"  x : [0..{width - 1}] init {start[0]};\n")
        f.write(f"  y : [0..{height - 1}] init {start[1]};\n")
        f.write(f"  steps : [0..{max_steps}] init 0;\n\n")

        action_map = {
            0: (0, -1),  # up
            1: (1, 0),   # right
            2: (0, 1),   # down
            3: (-1, 0)   # left
        }

        for (x, y), value in policy.items():
            if (x, y) == goal:
                continue

            # 判断 greedy 还是 stochastic 策略
            if isinstance(value, (int, np.integer)):
                # Greedy policy + noise
                action_probs = [noise / 3] * 4
                action_probs[value] = 1.0 - noise
            else:
                action_probs = list(value)

            transitions = []
            for action, prob in enumerate(action_probs):
                if prob == 0:
                    continue
                dx, dy = action_map[action]
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    transitions.append((prob, nx, ny))

            if transitions:
                total_prob = sum(p for p, _, _ in transitions)
                normed = [(p / total_prob, tx, ty) for (p, tx, ty) in transitions]

                trans_strs = []
                acc = 0.0
                for i, (p, tx, ty) in enumerate(normed):
                    if i < len(normed) - 1:
                        rounded_p = round(p, 2)
                        acc += rounded_p
                        trans_strs.append(
                            f"{rounded_p:.2f} : (x'={tx}) & (y'={ty}) & (steps'=steps+1)"
                        )
                    else:
                        last_p = round(1.0 - acc, 2)
                        trans_strs.append(
                            f"{last_p:.2f} : (x'={tx}) & (y'={ty}) & (steps'=steps+1)"
                        )

                f.write(f"  [] x={x} & y={y} & steps<{max_steps} -> {' + '.join(trans_strs)};\n")

        # 终止状态：达到最大步数后不再变化
        f.write(f"  [] steps={max_steps} -> (x'=x) & (y'=y) & (steps'=steps);\n")

        f.write("endmodule\n\n")

        # 标签
        f.write(f'label "goal" = x={goal[0]} & y={goal[1]};\n')
        if danger_list:
            danger_expr = " | ".join([f"(x={dx} & y={dy})" for dx, dy in danger_list])
            f.write(f'label "danger" = {danger_expr};\n')

        # reward 只在未超过最大步数时累计
        f.write('rewards "path"\n')
        f.write(f"  [] steps < {max_steps} : 1;\n")
        f.write("endrewards\n")
