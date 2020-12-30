import cv2
import os

# get all images paths
image_paths = os.listdir("testImages/")
for i in image_paths:
    if i.startswith('.'):
        image_paths.remove(i)

for i in image_paths:

    img = cv2.imread("testImages/" + i)

    img = cv2.resize(img,(250,250))
   
    cv2.imwrite("testImagesResized/" + i, img)
