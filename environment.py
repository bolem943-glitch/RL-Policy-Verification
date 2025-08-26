import random
class GridWorld:
    def __init__(self, size=5, start=(0, 0), goal=(4, 4), danger=None):
        self.size = size
        self.width = size
        self.height = size
        self.start = start
        self.goal = goal
        self.danger = danger or []
        self.action_space = [0, 1, 2, 3]
        self.action_dim = len(self.action_space)
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
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
        reward = -1

        if self.state in self.danger:
            reward += -10.0
            #done = True
        elif self.state == self.goal:
            reward += 100.0
            done = True
        elif self.state == (old_x, old_y):
            reward = -10.0  # ❗惩罚撞墙行为（hit the wall）


        return self.state, reward, done
