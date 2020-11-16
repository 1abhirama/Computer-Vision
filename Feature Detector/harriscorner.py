import cv2
import matplotlib.pyplot as plt 
import numpy as np

#HARRIS -> cv2.cornerHarris(img(grayscale & float32),size neighborhood,kernel sobel, konstanta harris(0.04,0.06)"semakin besar konstanta, semakin kecil yg kedetect")

#1. load image
img = cv2.imread('checkerboard_101.png')

#2. convert to greyscale dan float 32
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_gray = np.float32(img_gray)

#3. call function
harris_result = cv2.cornerHarris(img_gray, 2, 5, 0.04)

without_subpix = img.copy()
without_subpix[harris_result > 0.01*harris_result.max()] = [0,0,255]
without_subpix = cv2.cvtColor(without_subpix, cv2.COLOR_BGR2RGB)
plt.imshow(without_subpix)
plt.show()


