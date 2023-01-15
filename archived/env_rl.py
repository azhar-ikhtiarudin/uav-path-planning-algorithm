import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 40

HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 3000  # how often to play through env visually.

start_q_table = None # None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}

class Wall:
    def __init__(self, lx, y):
        self.lx = lx
        self.y = y
        # self.x = int(SIZE/2)
        self.xi = int(SIZE/2-lx/2)

class Blob:
    def __init__(self, x=False, y=False):
        if not x and not y:
            self.x = np.random.randint(0, SIZE)
            self.y = np.random.randint(0, SIZE)
        else:
            self.x = x
            self.y = y
        
    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y


        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


#FIRST START
# if start_q_table is None:
#     # initialize the q-table#
#     q_table = {}
#     for i in range(-SIZE+1, SIZE):
#         for ii in range(-SIZE+1, SIZE):
#             for iii in range(-SIZE+1, SIZE):
#                     for iiii in range(-SIZE+1, SIZE):
#                         q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(4)]

# else:
#     with open(start_q_table, "rb") as f:
#         q_table = pickle.load(f)

player = Blob()
food = Blob()
WallPosition = np.array([[-int(SIZE/2), int(SIZE/2)], [int(SIZE/2), int(SIZE/2)]])
print(WallPosition)


#LOOP
# episode_rewards = []
# for episode in range(HM_EPISODES):
#     player = Blob()
#     food = Blob()
#     wall_1 = Wall(int(SIZE/2), int(SIZE/2))
#     if episode % SHOW_EVERY == 0:
#         print(f"on #{episode}, epsilon is {epsilon}")
#         print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
#         show = True
#     else:
#         show = False
        
#     episode_reward = 0
    
        
        



env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue

#Red Box
# for i in range(40):
#     for ii in range(40):
#         if i == 0 or i == 39 or ii == 0 or ii == 39:
#             env[i][ii] = d[ENEMY_N]

for i in range(40)

img = Image.fromarray(env, 'RGB')
img = img.resize((900, 900), resample = Image.BOX)

cv2.imshow("image", np.array(img))  # show it!
cv2.waitKey(0)
cv2.destroyAllWindows()