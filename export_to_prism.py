def export_to_prism(policy, width, height, start, goal, danger_list, filename="policy_model.prism"):

    with open(filename, "w") as f:
        f.write("dtmc\n\n")
        f.write("module grid\n")
        f.write(f"  x : [0..{width - 1}] init {start[0]};\n")
        f.write(f"  y : [0..{height - 1}] init {start[1]};\n\n")

        for (x, y), action in policy.items():
            if (x, y) == goal:
                continue  # 目标状态不再移动
            #if (x, y) in danger_list:
             #   continue  # 陷阱状态不再移动

            if action == 0:     # up
                new_x, new_y = x, max(y - 1, 0)
            elif action == 1:   # right
                new_x, new_y = min(x + 1, width - 1), y
            elif action == 2:   # down
                new_x, new_y = x, min(y + 1, height - 1)
            elif action == 3:   # left
                new_x, new_y = max(x - 1, 0), y
            else:
                continue  # 忽略无效动作

            f.write(f"  [] x={x} & y={y} -> (x'={new_x}) & (y'={new_y});\n")

        f.write("endmodule\n\n")

        # 标记目标与危险区域
        f.write(f'label "goal" = x={goal[0]} & y={goal[1]};\n')
        if danger_list:
            danger_label = " | ".join([f"(x={dx} & y={dy})" for dx, dy in danger_list])
            f.write(f'\nlabel "danger" = {danger_label};\n')
        f.write(f'rewards "path"\n[] true : 1; \nendrewards')
