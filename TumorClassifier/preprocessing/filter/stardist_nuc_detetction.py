from stardist.models import StarDist2D

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois

import os

import shutil


from csbdeep.utils import normalize
import matplotlib.pyplot as plt






if __name__ == '__main__':
    pathIn =  r"C:\Users\felix\Desktop\neuro\thTestSmear\235\tiles"

    pathDec = r"C:\Users\felix\Desktop\neuro\thTestSmear\235\dec"
    pathAcc = r"C:\Users\felix\Desktop\neuro\thTestSmear\235\acc"


    # creates a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_he')

    lbl_cmap = random_label_cmap()


    for imageName in os.listdir(pathIn):
        imgPath = os.path.join(pathIn,imageName)
        
        img = plt.imread(imgPath)

        labels, details = model.predict_instances(normalize(img))

        numNucs = len(details['points'])

        print(imageName+ "   " + str(numNucs))

        if numNucs > 1:
            destPath = os.path.join(pathAcc,imageName)
        else:
            destPath = os.path.join(pathDec,imageName)
        shutil.copy(imgPath,destPath)








