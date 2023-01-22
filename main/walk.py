# Import Libraries
from model.DQNAgent import DQNAgent
from env.drone_env import DroneEnv 

import numpy as np
import tensorflow as tf
import time
import random
from tqdm import tqdm
import os

# Init variables and constants
#region
STATE_SIZE = 3
ACTION_SIZE = 9
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

if not os.path.isdir('images'):
    os.makedirs('images')
#endregion

# MAIN PROGRAM

# A. Initialize DQNAgent (include policy network, target network, and replay memory capacity)
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
env = DroneEnv()
obs = env.reset()

for step in range(10):
    print('now at step: ', step)
    env.saveImage(step)
    new_state, reward, done = env.step(1, obs)
    print(new_state, reward, done)