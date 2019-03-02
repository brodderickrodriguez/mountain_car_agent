import gym
from mountain_car import ReinforcementAlgorithm

gym.envs.register(
    id='MountainCarExtraLong-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=5000)

env = gym.make('MountainCarExtraLong-v0')
# env = gym.make('MountainCar-v0')


NUM_BINS = 10

NUM_PERCEPTRONS = 2  # cart position, cart velocity

NUM_STATES = NUM_BINS ** NUM_PERCEPTRONS  # 10^2 = 100

GAMMA = 0.99

ALPHA = 0.05

# ra = ReinforcementAlgorithm.QAlgorithm()
ra = ReinforcementAlgorithm.SARSA()
# ra = ReinforcementAlgorithm.EXPSARSA()
