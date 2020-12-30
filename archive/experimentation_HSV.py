import cv2
from sklearn.cluster import MiniBatchKMeans
import numpy as np

i = cv2.imread("testImages/ColorblindnessTest2.jpg")
HSV = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)

H,S,V=cv2.split(HSV)
# cv2.imshow("H Channel",H) # For A Channel (Here's what You need)

numClusters = 2
reshaped = H.reshape(H.shape[0] * H.shape[1],1)
minibatchkmeans = MiniBatchKMeans(n_clusters=numClusters, n_init=40, max_iter=500).fit(reshaped)

clustering = np.reshape(np.array(minibatchkmeans.labels_, dtype=np.uint8),
    (H.shape[0], H.shape[1]))
# Sort the cluster labels in order of the frequency with which they occur.
sortedLabels = sorted([n for n in range(numClusters)],
    key=lambda x: -np.sum(clustering == x))
kmeansImage = np.zeros(H.shape[:2], dtype=np.uint8)
for i, label in enumerate(sortedLabels):
    kmeansImage[clustering == label] = int((255) / (numClusters - 1)) * i

cv2.imshow("H Channel",kmeansImage)
# cv2.imshow("Image", edges)
cv2.waitKey(0)