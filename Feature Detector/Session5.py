# Corner Detection -> detecting corner
#1) Harris -> with subpix, without subpix
    # with subpix -> akurasi lebih tingga
#2) FAST -> Feature from segment test
#3) ORB -> Oriented fast and rotated brief

import cv2
import matplotlib.pyplot as plt 
import numpy as np

#HARRIS -> cv2.cornerHarris(img(grayscale & float32),size neighborhood,kernel sobel, konstanta harris(0.04,0.06)"semakin besar konstanta, semakin kecil yg kedetect")

#1. load image
img = cv2.imread("shape.png")
#cv2.imshow('image', img)
#cv2.waitKey(0)

#2. convert to greyscale dan float 32
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('images', img_gray)
#cv2.waitKey(0)
img_gray = np.float32(img_gray)

#3. call function
harris_result = cv2.cornerHarris(img_gray, 2, 5, 0.04)
#cv2.imshow('result', harris_result)
#cv2.waitKey(0)

harris_result2 = cv2.cornerHarris(img_gray, 2, 5, 0.06)
#cv2.imshow('result2', harris_result2)
#cv2.waitKey(0)

#without subpix 
without_subpix = img.copy()
without_subpix[harris_result > 0.01*harris_result.max()] = [0,0,255]
without_subpix = cv2.cvtColor(without_subpix, cv2.COLOR_BGR2RGB)
plt.imshow(without_subpix)
plt.show()

#with subpix
#cv2.cornerSubPix(image grayscale float32, centroid, size yg akan kita search(n,n) =>n*2+1 x n*2-1, size yang tidak dipakai (-1,-1), criteria) 
#hasilnya array of float corner

#1. pilih titik-titik yg akan di pakai -> threshold
# parameter 1 : image
# parameter 2 : value threshold
# parameter 3 : value color pixel
# parameter 4 : algoritma threshold
# return value -> retVal, threshold
_, thresh = cv2.threshold(harris_result, 0.01*harris_result.max(), 255, cv2.THRESH_BINARY)

#2. centroid : titik tengah dari pixel-pixel
#retVal, label, stats, centroid
thresh = np.uint8(thresh)
_, _, _, centroid = cv2.connectedComponentsWithStats(thresh)

#3. criteria : penanda untuk berhentinya algoritma subpix
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.001)
#TERM_CRITERIA_MAX_ITER : untuk nandain kita udah looping keberapa
#TERM_CRITERIA_EPS : untuk nandain perubahan value sebelumnya dengan sekarang
#parameter 2 : max iterasi
#parameter 3 : min value

#4. call function
# centroid -> float32
centroid = np.float32(centroid)
corner_subpix = cv2.cornerSubPix(img_gray, centroid, (2,2), (-1,-1), criteria)
print(corner_subpix)

with_subpix = img.copy()
corner_subpix = np.int16(corner_subpix) #convert ke integer karena mau akses index

for coor in corner_subpix:
    coor_x = coor[0]
    coor_y = coor[1]
    
    with_subpix[coor_y, coor_x] = [0,0,255]
    
with_subpix = cv2.cvtColor(with_subpix, cv2.COLOR_BGR2RGB)
plt.imshow(with_subpix)
plt.show()

##FAST

import os #kita mau looping isi file dari suatu directory

for filename in os.listdir("images"):
    #1. load image
    img = cv2.imread("images/" + filename)
    
    #2. buat object fast
    fast = cv2.FastFeatureDetector_create()
    
    #3. Detect corner -> cari keypoint
    keypoint = fast.detect(img)
    
    #4. drawKeypoints
        #parameter 1 : image
        #parameter 2 : keypoint
        #parameter 3 : output img
        #parameter 4 : color
    fast_result = img.copy()
    cv2.drawKeypoints(img,keypoint,fast_result,color = [255,0,0])
    cv2.imshow('fast', fast_result)
    cv2.waitKey(0)
    
##ORB

for filename in os.listdir("images"):
    #1. load image
    img = cv2.imread("images/" + filename)
    
    #2. buat object ORB
    img = cv2.ORB_create()
    
    #3. detect corner
    keypoint = orb.detect(img)
    
    #4. draw keypoints
    orb_result = img.copy()
    cv2.drawKeypoints(img, keypoint, orb_result, color=[0,255,0])
    cv2.imshow(filename, orb_result)
    cv2.waitKey(0)