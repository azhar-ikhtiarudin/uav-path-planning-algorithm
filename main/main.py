import numpy as np
import cv2
from PIL import Image
from env.drone_env import env, walls, agent_1

# define parameters
epsilon = 1.0
epsilon_min = 0.01
alpha = 0.5
alpha_min = 0.05
decay = 0.00001
EPISODES = 1000
SHOW_EVERY = 50


# ===================== Main Flow =====================
# Repeat for each episode:
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
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, 4)        # exploration
        else:
            action = 1  #np argmax                  # exploitation
        
        # 8. take action a, observe reward r and new state s'
        is_wall = env.is_wall(action) #kalau ada wall bakal true
        while is_wall: #selama True (ada wall), cari action lain
            action = np.random.randint(0, 8)
            is_wall = env.is_wall(action) #kalau false, berarti udah gak ada wall, lanjut next step

        next_state, reward, done = env.step(action)
        
        if SHOW_EVERY > 0 and episode % SHOW_EVERY == 0:
            env.render()
        
        current_state = next_state
        step += 1
    
    if epsilon >= epsilon_min:
            epsilon = epsilon*np.exp(-decay)
    
    if alpha >= alpha_min:
        alpha = alpha*np.exp(-decay)