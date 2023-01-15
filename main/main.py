import numpy as np
import cv2
from PIL import Image
from env.drone_env import env, walls, agent_1

# define parameters
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01


# ===================== Markov Decision Process Definition =====================
#define actions: 0 = up, 1 = down, 2 = left, 3 = right, ...; action ini ntar diubah jadi arah menurut method di class Drone
#define state: get_state() function ; ceritanya dapet sound level lalu dapet jarak lalu dapet state
#check if state is terminal: is_terminal() function ; ceritanya kalau udah di target, berarti terminal
#check it there is a wall: is_wall() function ; ngecek kalau ada wall, cari action lain

# ===================== Define Pseudofunctions =====================
def get_state():
    # get sound level from RSS sensor
    # use supervised machine learning to get distance from sound level
    # return distance
    pass

def is_wall():
    # if walls[agent.y+action.y,agent.x+action.x] == 1: 
    #    return True (berarti ada wall, cari random action lain)
    #    action = np.random.randint(0, action_space)            
    # else: return False (berarti tidak ada wall, lanjutkan)
    #    agent1.x += action.x
    #    agent1.y += action.y
    pass

def is_terminal():
    # if agent1.x == target.x and agent1.y == target.y:
    #   return True (berarti sudah sampai di target)
    pass


# ===================== Main Functions =====================

# 1. start from initial location and obtain associated state of that 
# particular location by sensing the RSS
x_0 = agent_1.x
y_0 = agent_1.y

# 2. Repeat for each step:
done = False
while not done:
    # 3,4 epsilon decay function
    if epsilon >= epsilon_min:
        epsilon *= epsilon_decay
    
    # 5,6 alpha decay function (?)
    
    # 7. select action based on epsilon greedy policy
    if np.random.rand() <= epsilon:
        action = np.random.randint(0, 4)        # exploration
    else:
        action = 1                              # exploitation
    
    # 8. take action a, observe reward r and new state s'
    next_state, reward, done, _ = env.step(action)