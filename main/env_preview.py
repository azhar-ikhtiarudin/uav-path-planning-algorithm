import numpy as np
import cv2
from PIL import Image
from env.drone_env import env


print(env.initial())
observation = env.initial()
# next_state, reward, done = env.step(1, observation)
# print(next_state, reward, done)
env.visualize()
env.render()