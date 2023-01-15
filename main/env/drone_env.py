import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

#Define Parameters
SIZE_Y = 40*2
SIZE_X = 60*2
ENV_COLOR = (20, 52, 89)
WALLS_COLOR = (77, 77, 234)
DRONE_COLOR = (234, 222, 53)
TARGET_COLOR = (132, 234, 53)

#Define Functions and Classes
def createLine(walls):
    #Full Horizontal Line
    # for i in range(SIZE_X):
    #     walls[int(SIZE_Y/2),int(i)] = 1
        
    #Full Vertical Line
    # for i in range(SIZE_Y):
        # walls[int(i), int(SIZE_X/2)] = 1

    #Segment 1
    for i in range(SIZE_Y//10, SIZE_Y//3):
        walls[i, SIZE_X//2] = 1

    #Segment 2
    for i in range(SIZE_X//2, SIZE_X//2+SIZE_X//6):
        walls[SIZE_Y//10, i] = 1

    #Segment 3
    for i in range(SIZE_Y//10, SIZE_Y//4):
        walls[i, SIZE_X//2+SIZE_X//6] = 1

    #Segment 4
    for i in range(SIZE_X//2-SIZE_X//6, SIZE_X//2+1):
        walls[SIZE_Y//3, i] = 1

    #Segment 5
    for i in range(SIZE_Y//3, SIZE_Y//3+SIZE_Y//4):
        walls[i, SIZE_X//2-SIZE_X//6] = 1

    #Segment 6
    for i in range(SIZE_X//2-SIZE_X//6, SIZE_X//2+SIZE_X//6):
        walls[SIZE_Y//3+SIZE_Y//4, i] = 1
        
    #Segment 7
    for i in range(SIZE_Y//3+SIZE_Y//4, SIZE_Y//3+SIZE_Y//4+SIZE_Y//5):
        walls[i, SIZE_X//2+SIZE_X//6] = 1

    #Segment 8
    for i in range(SIZE_X//2+SIZE_X//6, SIZE_X//2+SIZE_X//6+SIZE_X//6):
        walls[SIZE_Y//3+SIZE_Y//4+SIZE_Y//5, i] = 1

    #Segment 9
    for i in range(SIZE_Y-SIZE_Y//3, SIZE_Y):
        walls[i, SIZE_X//2] = 1
        
    #Segment 10
    for i in range(SIZE_Y-SIZE_Y//4, SIZE_Y):
        walls[i, SIZE_X//2-SIZE_X//4] = 1

    #Segment 11
    for i in range(0, SIZE_X//6):
        walls[SIZE_Y//5, i] = 1

    #Segment 12
    for i in range(SIZE_Y//5, SIZE_Y//5+SIZE_Y//5):
        walls[i, SIZE_X//6] = 1

    #Segment 13
    for i in range(0, SIZE_X//6):
        walls[SIZE_Y//5+SIZE_Y//3, i] = 1
        
    #Segment 14
    for i in range(0, SIZE_Y//3):
        walls[i, SIZE_X-SIZE_X//6] = 1

    #Segment 15
    for i in range(SIZE_X-SIZE_X//10, SIZE_X):
        walls[SIZE_Y//3, i] = 1

    #Segment 16
    for i in range(SIZE_X-SIZE_X//9, SIZE_X):
        walls[SIZE_Y//2+SIZE_Y//10, i] = 1

    #Segment 17
    for i in range(SIZE_Y//2+SIZE_Y//10, SIZE_Y//2+SIZE_Y//10+SIZE_Y//4):
        walls[i, SIZE_X-SIZE_X//9] = 1
        
    #Segment 18
    for i in range(SIZE_Y//2+SIZE_Y//10+SIZE_Y//3, SIZE_Y):
        walls[i, SIZE_X-SIZE_X//9] = 1

class EnvObject:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.x}, {self.y}"
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
class Drone(EnvObject):
    def __init__(self, x, y):
        super().__init__(x, y)
        
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
        elif self.x > SIZE_Y-1:
            self.x = SIZE_Y-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE_Y-1:
            self.y = SIZE_Y-1

class Target(EnvObject):
    def __init__(self, x, y):
        super().__init__(x, y)

#Create Objects
agent_1 = Drone(SIZE_X-SIZE_X//11, SIZE_Y//10)
agent_2 = Drone(SIZE_X-SIZE_X//20, SIZE_Y//2-SIZE_Y//20)
agent_3 = Drone(SIZE_X-SIZE_X//13, SIZE_Y-SIZE_Y//11)
target = Target(SIZE_X//10, SIZE_Y//2-SIZE_Y//20)

#Create Environment and Walls
env = np.zeros((SIZE_Y, SIZE_X, 3), dtype=np.uint8)  # starts an rbg of our size
walls = np.zeros((SIZE_Y, SIZE_X), dtype=np.uint8)
createLine(walls)

#Color the Walls and Environment
for i in range(SIZE_Y):
    for j in range(SIZE_X):
        if walls[i][j] == 1:
            env[i][j] = WALLS_COLOR
        else:
            env[i][j] = ENV_COLOR

#Color the Objects  
env[agent_1.y][agent_1.x] = DRONE_COLOR
env[agent_2.y][agent_2.x] = DRONE_COLOR
env[agent_3.y][agent_3.x] = DRONE_COLOR
env[target.y][target.x] = TARGET_COLOR

