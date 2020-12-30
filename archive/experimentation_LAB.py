import cv2
from sklearn.cluster import MiniBatchKMeans
import numpy as np
i = cv2.imread("testImages/ColorblindnessTest.jpg")
lab = cv2.cvtColor(i, cv2.COLOR_BGR2LAB)
L,A,B=cv2.split(lab)

A = cv2.resize(A,(300,300))

img = cv2.medianBlur(A,15)
img = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)
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
kernel = np.ones((5,5), np.uint8)
img = cv2.erode(img, kernel, iterations = 1)

cv2.imshow("Image", img)
cv2.waitKey(0)

"""

cv2.imshow("Image", img)
cv2.waitKey(0)

"""


"""
# cv2.imshow("A_Channel",A) # For A Channel (Here's what You need)
numClusters = 2
reshaped = A.reshape(A.shape[0] * A.shape[1],1)
minibatchkmeans = MiniBatchKMeans(n_clusters=numClusters, n_init=40, max_iter=500).fit(reshaped)

clustering = np.reshape(np.array(minibatchkmeans.labels_, dtype=np.uint8),
    (A.shape[0], A.shape[1]))
# Sort the cluster labels in order of the frequency with which they occur.
sortedLabels = sorted([n for n in range(numClusters)],
    key=lambda x: -np.sum(clustering == x))
kmeansImage = np.zeros(A.shape[:2], dtype=np.uint8)
for i, label in enumerate(sortedLabels):
    kmeansImage[clustering == label] = int((255) / (numClusters - 1)) * i

kernel = np.ones((5,5), np.uint8)
img = cv2.morphologyEx(A, cv2.MORPH_CLOSE, kernel)
# apply morphology open
kernel = np.ones((5,5), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# erosion (to make it thinner)
kernel = np.ones((5,5), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)

cv2.imshow("Image", A)
cv2.imshow("Image", img)
cv2.waitKey(0)

#kernel = np.ones((3,3),np.uint8)

# kernel = np.ones((5,5),np.uint8)
# erosion = cv.erode(img,kernel,iterations = 1)
cv2.imshow("Image", img)
cv2.waitKey(0)
"""