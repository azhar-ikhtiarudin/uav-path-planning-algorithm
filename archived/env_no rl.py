import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE_X = 100
SIZE_Y = 25
epsilon = 0.9
start_q_table = None # None or Filename
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 3000  # how often to play through env visually.
LEARNING_RATE = 0.1
DISCOUNT = 0.95
PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE_X)
        self.y = np.random.randint(0, SIZE_X)

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
        elif self.x > SIZE_X-1:
            self.x = SIZE_X-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE_X-1:
            self.y = SIZE_X-1

player = Blob()
food = Blob()
wall_1 = Wall(int(SIZE_X/2), int(SIZE_X/2))


env = np.zeros((SIZE_X, SIZE_X, 3), dtype=np.uint8)  # starts an rbg of our size
env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue

for i in range(wall_1.lx):
    env[wall_1.xi][wall_1.y] = d[ENEMY_N]
    wall_1.xi += 1

img = Image.fromarray(env, 'RGB')
img = img.resize((900, 900), resample = Image.BOX)

q_table = np.zeros((SIZE_X, SIZE_Y), dtype=np.int32)


cv2.imshow("image", np.array(img))  # show it!
cv2.waitKey(0)
cv2.destroyAllWindows()