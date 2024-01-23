from stardist.models import StarDist2D

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois

from stardist.plot import render_label
from csbdeep.utils import normalize

import os

import shutil


from csbdeep.utils import normalize
import matplotlib.pyplot as plt

import cv2






if __name__ == '__main__':
    pathIn =  r"E:\he_test"
    
    out = r"E:\he_output"
    out2 = r"E:\chan"

    pathDec = r"C:\Users\felix\Desktop\neuro\thTestSmear\235\dec"
    pathAcc = r"C:\Users\felix\Desktop\neuro\thTestSmear\235\acc"


    # creates a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_he')

    lbl_cmap = random_label_cmap()


    for imageName in os.listdir(pathIn):
        imgPath = os.path.join(pathIn,imageName)
        
        img = plt.imread(imgPath)
        
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]
        
        grey =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        safePath2 = os.path.join(out2, "red_"+imageName)
        cv2.imwrite(safePath2 ,red)
        
        safePath2 = os.path.join(out2, "green_"+imageName)
        cv2.imwrite(safePath2 ,green)
        
        safePath2 = os.path.join(out2, "blue_"+imageName)
        cv2.imwrite(safePath2 ,blue)
        
        safePath2 = os.path.join(out2, "grey_"+imageName)
        cv2.imwrite(safePath2 ,grey)
        
        #tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        
        
        
        
        

        labels, details = model.predict_instances(normalize(img))
        
        image = render_label(labels, img=img)
        
        
        
        plt.figure(figsize=(15,15))
        
        safePath2 = os.path.join(out, imageName)
        
        plt.imshow(image)
        plt.axis("off")
        plt.title("prediction + input overlay")
        plt.savefig(safePath2)
        plt.close()

        numNucs = len(details['points'])

        print(imageName+ "   " + str(numNucs))

        