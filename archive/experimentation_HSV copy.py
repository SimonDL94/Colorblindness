import cv2
from sklearn.cluster import MiniBatchKMeans
import numpy as np

i = cv2.imread("testImages/ColorblindnessTest2.jpg")
HSV = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)

H,S,V=cv2.split(HSV)
# cv2.imshow("H Channel",H) # For A Channel (Here's what You need)

def apply(H,K):
        vectorized = H.reshape((-1,3))
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts=10
        ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((H.shape))        
        return result_image


result = apply(H,2)
cv2.imshow("H Channel",result)
# cv2.imshow("Image", edges)
cv2.waitKey(0)