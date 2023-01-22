import numpy as np
import cv2
from PIL import Image
from env.drone_env import DroneEnv

env = DroneEnv()
obs = env.reset()
env.saveImage(7)
# print(status)

for step in range(200):
    