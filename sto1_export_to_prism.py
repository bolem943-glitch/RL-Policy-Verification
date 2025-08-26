def export_to_prism(policy, width, height, start, goal, danger_list, noise_prob=0.36, filename="stochastic_policy_model.prism"):
    with open(filename, "w") as f:
        f.write("dtmc\n\n")
        f.write("module grid\n")
        f.write(f"  x : [0..{width - 1}] init {start[0]};\n")
        f.write(f"  y : [0..{height - 1}] init {start[1]};\n\n")

        # 动作映射函数
        def move(a, x, y):
            if a == 0:
                return x, max(y - 1, 0)  # up
            elif a == 1:
                return min(x + 1, width - 1), y  # right
            elif a == 2:
                return x, min(y + 1, height - 1)  # down
            elif a == 3:
                return max(x - 1, 0), y  # left

        for (x, y), best_action in policy.items():
            if (x, y) == goal:
                continue  # 到达目标则不再移动

            # 主策略动作
            main_prob = 1 - noise_prob
            main_x, main_y = move(best_action, x, y)
            transitions = [(main_prob, main_x, main_y)]

            # 处理随机动作（排除主动作）
            other_actions = [a for a in range(4) if a != best_action]
            valid_random_moves = []
            for a in other_actions:
                tx, ty = move(a, x, y)
                if (tx, ty) != (x, y):  # 过滤撞墙动作
                    valid_random_moves.append((tx, ty))

            if valid_random_moves:
                redistributed_prob = noise_prob / len(valid_random_moves)
                for tx, ty in valid_random_moves:
                    transitions.append((redistributed_prob, tx, ty))
            else:
                # 若全是撞墙，则将 noise_prob 概率分给当前位置（stay）
                transitions.append((noise_prob, x, y))

            # 直接写入，不合并重复位置，保留多个分支
            trans_strs = [f"{p:.2f} : (x'={tx}) & (y'={ty})" for (p, tx, ty) in transitions]
            f.write(f"  [] x={x} & y={y} -> {' + '.join(trans_strs)};\n")

        f.write("endmodule\n\n")

        # 标签：目标 + 陷阱
        f.write(f'label "goal" = x={goal[0]} & y={goal[1]};\n')
        if danger_list:
            danger_expr = " | ".join([f"(x={dx} & y={dy})" for dx, dy in danger_list])
            f.write(f'label "danger" = {danger_expr};\n')

        # 可选奖励结构（路径惩罚）
        f.write('rewards "path"\n  [] true : 1;\nendrewards\n')
