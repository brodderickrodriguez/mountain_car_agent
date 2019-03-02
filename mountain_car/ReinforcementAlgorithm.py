import numpy as np
from mountain_car import Configuration as Config
import operator
import logging


class ReinforcementAlgorithm:
    algorithm_sequence = 0

    def next_action(self, q, next_state, epsilon, is_training=True):
        if np.random.uniform() < epsilon and is_training:
            # logging.info('random action selected for state {s} given epsilon={e}'.format(s=next_state, e=epsilon))
            return int(np.round(np.random.uniform() * (len(q[next_state]) - 1)))
        else:
            return max(q[next_state].items(), key=operator.itemgetter(1))[0]

    def get_new_q_value(self, q, state, action, next_reward, next_state, next_action):
        pass


class QAlgorithm(ReinforcementAlgorithm):
    algorithm_sequence = 1

    @staticmethod
    def value(q, state):
        return max(q[state].items(), key=operator.itemgetter(1))[1]

    def get_new_q_value(self, q, state, action, next_reward, next_state, next_action):
        return Config.ALPHA * (next_reward + Config.GAMMA * self.value(q, next_state) - q[state][action])


class SARSA(ReinforcementAlgorithm):
    algorithm_sequence = 2

    def get_new_q_value(self, q, state, action, next_reward, next_state, next_action):
        return Config.ALPHA * (next_reward + Config.GAMMA * q[next_state][next_action] - q[state][action])


class EXPSARSA(SARSA):
    algorithm_sequence = 3

    @staticmethod
    def weighted_sum(q, next_state, next_action):
        d = sum([v for _, v in q[next_state].items()])
        d = 1 if d == 0 else d
        # d = q[next_state][next_action]
        return d

    def get_new_q_value(self, q, state, action, next_reward, next_state, next_action):
        return Config.ALPHA * (next_reward + Config.GAMMA * self.weighted_sum(q, next_state, next_action) - q[state][action])
