import numpy as np
import cv2
from PIL import Image
from env.drone_env import env 
from model.dqn import DQNAgent
import os

# define parameters
epsilon = 1.0
epsilon_min = 0.01
alpha = 0.5
alpha_min = 0.05
decay = 0.00001
state_size = 3
action_size = 9
batch_size = 32
EPISODES = 1000
SHOW_EVERY = 50
output_dir = 'model_output/uav-rl/'

# ===================== Main Flow =====================
# Repeat for each episode:
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
agent = DQNAgent(state_size, action_size)
for episode in range(EPISODES):
    episode_reward = 0
    step = 1
    
    # 1. start from initial location and obtain associated state of that 
    # particular location by sensing the RSS
    current_state = env.initial()

    # 2. Repeat for each step:
    done = False
    while not done:
        # 7. select action based on epsilon greedy policy
        # if np.random.rand() <= epsilon:
        #     action = np.random.randint(0, 4)        # exploration
        # else:
        #     action = model.predict(current_state)
            # action = 1  #np argmax                  # exploitation
        
        action = agent.act(current_state)
        
        # 8. take action a, observe reward r and new state s'
        is_wall = env.is_wall(action) #kalau ada wall bakal true
        while is_wall: #selama True (ada wall), cari action lain
            action = np.random.randint(0, 8)
            is_wall = env.is_wall(action) #kalau false, berarti udah gak ada wall, lanjut next step

        next_state, reward, done = env.step(action, current_state)
        
        next_state = np.reshape(current_state, [1, state_size])
        
        agent.remember(current_state, action, reward, next_state, done)
        
        if SHOW_EVERY > 0 and episode % SHOW_EVERY == 0:
            env.render()
        
        current_state = next_state
        step += 1
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size )
        
        if episode % 50 == 0:
            agent.save(output_dir + "weights_" + '{:04d}'.format(episode) + '.hdf5' )
            

    if epsilon >= epsilon_min:
        epsilon = epsilon*np.exp(-decay)
    
    if alpha >= alpha_min:
        alpha = alpha*np.exp(-decay)