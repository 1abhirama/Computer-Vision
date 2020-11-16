# Feature Detection, FAST, ORB, SURF.
# Feature Accelarate Segment Test (FAST)
# Oriented Fast and Rotate Brief (ORB)
# Speeded-up Robust Feature (SURF)

import cv2
import matplotlib.pyplot as plt

#img = cv2.imread('images/apple.jpg')

#FAST
#fast = cv2.FastFeatureDetector_create()
#keypoint = fast.detect(img)

#fast_result = cv2.drawKeypoints(img, keypoint, None, color=[0,255,0])

#plt.subplot(1,3,1)
#plt.imshow(fast_result)
#plt.title('FAST')
#plt.xticks([])
#plt.yticks([])

#ORB
#orb = cv2.ORB_create()
#keypoint_orb = orb.detect(img)

#orb_result = cv2.drawKeypoints(img, keypoint_orb, None,color=[0,255,0])

#plt.subplot(1,3,1)
#plt.imshow(orb_result)
#plt.title('ORB')
#plt.xticks([])
#plt.yticks([])


#SURF

img_obj = cv2.imread('images/object.jpg',0)
img_scene = cv2.imread('images/scene.jpg',0)

# 1. Get all keypoints and descriptor

surf = cv2.xfeatures2d.SURF_create()

#output = keypoint, descriptor
keypoint_obj, descriptor_obj = surf.detectAndCompute(img_obj, None)
keypoint_scene, descriptor_scene = surf.detectAndCompute(img_scene, None)

# 2. Matching
# K-D Tree and Knn Match

FLANN_INDEX_KDTREE = 0
algo_param = dict(algorithm=FLANN_INDEX_KDTREE)
search_param = dict(checks=100)

flann = cv2.FlannBasedMatcher(algo_param, search_param)

matches = flann.knnMatch(descriptor_obj, descriptor_scene, k=2)

# Create Masking matches

match_mask = []
for i in range(len(matches)):
    match_mask.append([0,0])

# M = First best match
# N = Second best match

for idx, (m,n) in enumerate(matches):
    #Lowe's Paper
    if m.distance < 0.7 * n.distance:
        # m = ambil n = kosongin
        match_mask[idx] = [1,0]

# 3. Draw Matching Points
draw_result = cv2.drawMatchesKnn(img_obj, keypoint_obj, img_scene, keypoint_scene, matches, None, matchColor=[0,255,0], singlePointColor=[0,255,0])

plt.subplot(1,1,1)
plt.imshow(draw_result)
plt.xticks([])
plt.yticks([])
plt.show()

# buat looping folder images yg banyak isinya, import os dulu baru bikin loopingannya -> for i, filename in enumerate(os.listdir('images/')):


