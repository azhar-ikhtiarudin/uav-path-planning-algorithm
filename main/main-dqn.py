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
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save

# Environment settings
EPISODES = 20

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

for episode in range(EPISODES):
    os.makedirs('images/episode_' + str(episode+1))
#endregion

# MAIN PROGRAM

# A. Initialize DQNAgent (include policy network, target network, and replay memory capacity)
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
env = DroneEnv()

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'): #tqdms is a progress bar
    
    # // Update tensorboard step every episode
    agent.tensorboard.step = episode

    # // Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1
    
    # 1. Initialize the starting state
    current_state = env.reset()
    
    # 2. Reset flag and start iterating until episode ends
    done = False
    while not done:
        # a. Explore Exploit Tradeoff
        # b. Execute the action in the environment
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, ACTION_SIZE)
            # action = np.argmax(agent.get_qs(current_state))
        
        # c. Observe reward and next state
        new_state, reward, done = env.step(action, current_state)
        episode_reward += reward
        
        # // Render
        # if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
        #     env.render()
            
        # d. Store experience in replay memory
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        
        # e. Train the agent
        agent.train(done, step)
        current_state = new_state
        env.saveImage(episode, step)
        step += 1
    
    # 3. Tensorboard, append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    
    print('episode: ', episode, 'epsilon: ', epsilon, 'episode_reward: ', episode_reward)
        
        
        
        