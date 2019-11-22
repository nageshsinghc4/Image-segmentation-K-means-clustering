#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:39:21 2019

@author: nageshsinghchauhan
"""
#SImple image segmentation using K-Means clustering algo

#color clustering
#Image segmentation from video using OpenCV and K-means clustering
import numpy as np
import cv2
import matplotlib.pyplot as plt
original_image = cv2.imread("/Users/nageshsinghchauhan/Desktop/images/p.jpg")
# Converting from BGR Colours Space to HSV
img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
vectorized = img.reshape((-1,3))
# convert to np.float32
vectorized = np.float32(vectorized)
# Here we are applying k-means clustering so that the pixels around a colour are consistent and gave same BGR/HSV values
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# We are going to cluster with k = 2, because the image will have just two colours ,a white background and the colour of the patch
K = 4
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
# Now convert back into uint8
#now we have to access the labels to regenerate the clustered image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
#res2 is the result of the frame which has undergone k-means clustering
figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(res2)
plt.title('K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()

#canny edge detection
edges = cv2.Canny(img,100,200)
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

"""
#video
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture("/Users/nageshsinghchauhan/Documents/bb.mkv")

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video  file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # Display the resulting frame
    cv2.imshow('Frame', frame)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  # Break the loop
  else:
    break
# When everything done, release
# the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
"""




#plotting an Image in 3D color space.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

#read image
img = cv2.imread("/Users/nageshsinghchauhan/Documents/bb.mkv")

#convert from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#get rgb values from image to 1D array
r, g, b = cv2.split(img)
r = r.flatten()
g = g.flatten()
b = b.flatten()

#plotting
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(r, g, b)
plt.show()
