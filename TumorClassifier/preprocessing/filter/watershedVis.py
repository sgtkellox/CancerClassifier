import numpy as np 
import pandas as pd
import os
import pathlib
import imageio
import cv2
import skimage

im = cv2.imread(r"C:\Users\felix\Desktop\neuro\thTestSmear\235\accWaterShed\A2-N17-1152Q_10500_31500.jpg")#Read Images


im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
im1=im

#Plotting data for comparision of grayscale versus original image
import matplotlib.pyplot as plt


im_blur=cv2.GaussianBlur(im_gray,(5,5),0)


#Using Watershed
ret,th = cv2.threshold(im_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(th,cmap='gray')
plt.axis("off")
plt.show()

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel, iterations = 2)
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(opening,cmap='gray')
plt.axis("off")
plt.show()

sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.005*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(markers,cmap='jet')
plt.axis("off")
plt.show()

markers = cv2.watershed(im1,markers)
im1[markers == -1] = [255,0,0]
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(im1,cmap='gray')
plt.axis("off")
plt.show()

fig=plt.figure(figsize=(20, 20), dpi= 80, facecolor='w', edgecolor='k')
plt.axis("off")
plt.subplot(121)
plt.imshow(im1)
plt.axis("off")
plt.subplot(122)
plt.imshow(markers,cmap='gray')
plt.axis("off")
plt.show()

mask = np.where(markers > sure_fg, 1, 0)
# Make sure the larger portion of the mask is considered background
if np.sum(mask==0) < np.sum(mask==1):
    mask = np.where(mask, 0, 1)

from scipy import ndimage
labels, nlabels = ndimage.label(mask)
# Regenerate the labels
label_arrays = []
for label_num in range(1, nlabels+1):
    label_mask = np.where(labels == label_num, 1, 0)
    label_arrays.append(label_mask)
    
print('There are {} separate components / objects detected.'.format(nlabels))

for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
    cell = markers[label_coords]
    
    # Check if the label size is too small
    if np.product(cell.shape) < 1500: 
        #print('Label {} is too small! Setting to 0.'.format(label_ind))
        mask = np.where(labels==label_ind+1, 0, mask)

# Regenerate the labels
labels, nlabels = ndimage.label(mask)

label_arrays = []
for label_num in range(1, nlabels+1):
    label_mask = np.where(labels == label_num, 1, 0)
    label_arrays.append(label_mask)
    
print('There are now {} separate components / objects detected.'.format(nlabels))




