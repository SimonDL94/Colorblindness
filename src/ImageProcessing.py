import cv2
import numpy as np

class ImageProcessing():
    def read(self, image_path):
        image = cv2.imread(image_path)
        return image

    def apply_BGR2LAB(self, img, type = "LAB"):
        LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return LAB
    
    def apply_contrast(self, img, contrast = 0):
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            contrasted = cv2.addWeighted(img.copy(), alpha_c, img.copy(), 0, gamma_c)
            return contrasted

    def apply_processing_colorblind(self, img, P_CONTRAST = 100, n_loops = 5, gaussianblur_size = (3,3), medianblur_size = 15, resize = (28,28)):

        img = self.apply_contrast(img, P_CONTRAST)

        for k in range(n_loops):
            img = cv2.GaussianBlur(img, gaussianblur_size, cv2.BORDER_DEFAULT)

        img = cv2.medianBlur(img, medianblur_size)
        
        # thresholding to get binary values
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        # apply kernel with morphologyEx
        kernel = np.ones((8,8),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        img = cv2.resize(img,resize)

        return img