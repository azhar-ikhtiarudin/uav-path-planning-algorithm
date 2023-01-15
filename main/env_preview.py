import numpy as np
import cv2
from PIL import Image
from env.drone_env import env

#Show Environment
img = Image.fromarray(env, 'RGB')
img = img.resize((1200, 800), resample = Image.Resampling.BOX)
cv2.imshow("image", np.array(img))  # show it!
cv2.waitKey(0)
cv2.destroyAllWindows()