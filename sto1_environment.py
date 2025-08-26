import random

class GridWorld:
    def __init__(self, size=5, start=(0, 0), goal=(4, 4), danger=None):
        self.size = size
        self.width = size
        self.height = size
        self.start = start
        self.goal = goal
        self.danger = danger or []
        self.action_space = [0, 1, 2, 3]  # 上右下左
        self.action_dim = len(self.action_space)
        self.reset()

        # 动作 → 位移映射，用于 move 函数或导出
        self.action_map = {
            0: (0, -1),  # up
            1: (1, 0),   # right
            2: (0, 1),   # down
            3: (-1, 0)   # left
        }

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        dx, dy = self.action_map[action]
        new_x = max(0, min(self.width - 1, x + dx))
        new_y = max(0, min(self.height - 1, y + dy))

        old_state = self.state
        self.state = (new_x, new_y)
        done = False
        reward = -1  # base cost for each step

        if self.state in self.danger:
            reward += -1.0  # penalty for danger (can be more severe)
        elif self.state == self.goal:
            reward += 10.0
            done = True
        elif self.state == old_state:
            reward = -10.0  # hit wall penalty

        return self.state, reward, done

    def get_valid_actions(self, state=None):
        """可选：返回不会撞墙的合法动作，用于更细粒度控制策略"""
        if state is None:
            state = self.state
        x, y = state
        valid = []
        for a, (dx, dy) in self.action_map.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                valid.append(a)
        return valid
