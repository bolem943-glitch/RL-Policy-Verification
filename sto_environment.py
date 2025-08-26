import random
class GridWorld:
    def __init__(self, size=5, start=(0, 0), goal=(4, 4), danger=None, stochastic=False, noise_prob=0.42):
        self.size = size
        self.width = size
        self.height = size
        self.start = start
        self.goal = goal
        self.danger = danger or []
        self.stochastic = stochastic  # parameter for opening stochastic
        self.noise_prob = noise_prob  # the probability of stochastic
        self.action_space = [0, 1, 2, 3]  # 上右下左
        self.action_dim = len(self.action_space)
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        if self.stochastic and random.random() < self.noise_prob:
            action = random.choice(self.action_space)

        x, y = self.state
        old_x, old_y = self.state

        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # right
            x = min(self.size - 1, x + 1)
        elif action == 2:  # down
            y = min(self.size - 1, y + 1)
        elif action == 3:  # left
            x = max(0, x - 1)

        self.state = (x, y)
        done = False

        if self.state == self.goal:
            reward = 100  # ✅ 目标奖励大一些
            done = True
        elif self.state in self.danger:
            reward = -10  # ✅ 陷阱惩罚加大
            done = True
        elif self.state == (old_x, old_y):
            reward = -10  # ✅ 撞墙惩罚加大
        else:
            reward = -1  # 每步惩罚

        return self.state, reward, done

