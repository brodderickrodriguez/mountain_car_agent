import numpy as np


class RunIteration:
    def __init__(self, reward=np.NINF, step_count=np.Inf, q_func=None, gamma=None, alpha=None):
        self.reward, self.step_count, self.q_func = reward, step_count, q_func
        self.gamma = gamma
        self.alpha = alpha

    def print(self):
        print('steps: {s}, reward: {r}'.format(s=self.step_count, r=self.reward))
