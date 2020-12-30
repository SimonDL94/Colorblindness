import cv2
from sklearn.cluster import MiniBatchKMeans
import numpy as np

i = cv2.imread("testImages/ColorblindnessTest.jpg")
YCRCB = cv2.cvtColor(i, cv2.COLOR_BGR2YCR_CB)

Y,Cr,Cb=cv2.split(YCRCB)

img = cv2.medianBlur(Cr,15)
ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
# apply morphology close
kernel = np.ones((5,5), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# apply morphology open
kernel = np.ones((5,5), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# erosion (to make it thinner)
kernel = np.ones((10,10), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.medianBlur(img,11)


cv2.imshow("Image", img)
cv2.waitKey(0)
