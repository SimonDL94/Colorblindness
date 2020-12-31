from src.ImageProcessing import ImageProcessing
import os
import cv2 
import numpy as np
from keras.models import load_model

#### loading last trained model
model = load_model("src/CNN")

#### reading Image & Preprocessing
ip = ImageProcessing()

img = ip.read("PathToTestImage")

img = cv2.resize(img,(300,300))

LAB = ip.apply_BGR2LAB(img, type = "LAB")

# only keeping the A channel representing the red-green coloring
_, A, _ = cv2.split(LAB)

img = ip.apply_processing_colorblind(A, 
        P_CONTRAST = 100
        , n_loops = 5
        , gaussianblur_size = (3,3)
        , medianblur_size = 15
        , resize = (28,28))

#### reshaping and predict with trained model

# reshaping to fit
img = np.reshape(img, (1, 28, 28, 1))

# scaling
img = (img - 0) / (255 - 0)

# make prediction
pred = model.predict(img)

# predicting the number for each test
number = pred.argmax()

print(number)

