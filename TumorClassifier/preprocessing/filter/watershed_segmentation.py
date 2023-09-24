import numpy as np 
import pandas as pd
import os
import pathlib
import imageio
import cv2
import skimage

import matplotlib.pyplot as plt

from scipy import ndimage

import shutil





def isOverThreshold(path,threshold):
    im = cv2.imread(path)

    #im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im_gray = im[:,:,0]
    im1=im




    im_blur=cv2.GaussianBlur(im_gray,(5,5),0)

    ret,th = cv2.threshold(im_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel, iterations = 2)


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

    markers = cv2.watershed(im1,markers)
    im1[markers == -1] = [255,0,0]


    mask = np.where(markers > sure_fg, 1, 0)

    # Make sure the larger portion of the mask is considered background

    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)

    labels, nlabels = ndimage.label(mask)

    # Regenerate the labels
    label_arrays = []
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        label_arrays.append(label_mask)
    
    #print('There are {} separate components / objects detected.'.format(nlabels))

    for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
        cell = markers[label_coords]
    
        # Check if the label size is too small
        if np.product(cell.shape) < 2000: 
            #print('Label {} is too small! Setting to 0.'.format(label_ind))
            mask = np.where(labels==label_ind+1, 0, mask)

    # Regenerate the labels
    labels, nlabels = ndimage.label(mask)

    label_arrays = []
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        label_arrays.append(label_mask)
    
    print('There are now {} separate components / objects detected.'.format(nlabels))

    if nlabels>=threshold:
        print("th "+ str(threshold) +" labels :" + str(nlabels) + "   accepted")
        return True
    else:
        print("th "+ str(threshold) +" labels :" + str(nlabels) + "   declined")
        return False


def filterNuclei(threshold, inPath, accPath, decPath):
    
    for imageName in os.listdir(inPath):
        imagePath = os.path.join(inPath,imageName)
        if isOverThreshold(imagePath,threshold):
            destPath = os.path.join(accPath,imageName)
        else:
            destPath = os.path.join(decPath,imageName)
        shutil.copy(imagePath,destPath)

if __name__ == '__main__':

    threshold = 5

    inPath = r"C:\Users\felix\Desktop\neuro\thTestSmear\235\tiles"

    accPath = r"C:\Users\felix\Desktop\neuro\thTestSmear\235\accWaterShed"

    decPath = r"C:\Users\felix\Desktop\neuro\thTestSmear\235\decWaterShed"

    filterNuclei(threshold,inPath, accPath, decPath)



        

    






