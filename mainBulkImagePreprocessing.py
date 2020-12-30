from src.ImageProcessing import ImageProcessing

import os
import cv2 
import numpy as np
from keras.models import load_model

# loading last trained model
model = load_model("src/CNN")

# get all images paths
image_paths = os.listdir("testImages/")
for i in image_paths:
    if i.startswith('.'):
        image_paths.remove(i)

ip = ImageProcessing()

for i in image_paths:

    img = ip.read("testImages/" + i)

    img = cv2.resize(img,(300,300))

    LAB = ip.apply_BGR2LAB(img, type = "LAB")

    # only keeping the A channel representing the red-green coloring
    _, A, _ = cv2.split(LAB)

    img = ip.apply_processing_colorblind(A, 
        P_CONTRAST = 100
        , n_loops = 5
        , gaussianblur_size = (3,3)
        , medianblur_size = 15
        # set output images to 300 x 300 size
        , resize = (300,300))
   
    cv2.imwrite("testImagesProcessed/" + i, img)
