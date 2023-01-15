import gym
import numpy as np

env = gym.make("CartPole-v1")

env.reset()

print(gym.__version__)
print(np.__version__)

while True:
    env.render()
