import numpy as np
import matplotlib.pyplot as plt

import cv2

from pathml.core import HESlide
from pathml.preprocessing import StainNormalizationHE
import os


if __name__ == '__main__':
     path = r"D:\kyroSplit"

     for file in os.listdir(path):
        d = os.path.join(path, file)
        copiedFiles = os.listdir(os.path.join(r"C:\Users\felix\Desktop\neuro\augmentOutput",file,"macenko","hematoxylin"))
        if os.path.isdir(d):
            
            testImgs = os.listdir(d)
            
            for testImg in testImgs:
                if not testImg.endswith(".jpg"):
                    continue
                if testImg in copiedFiles:
                    print("file " + testImg+ " allready copied")
                    continue
                imgPath = os.path.join(d,testImg)
                image = cv2.imread(imgPath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                for i, method in enumerate(["macenko", "vahadane"]):
                    for j, target in enumerate(["normalize", "hematoxylin", "eosin"]):
                        # initialize stain normalization object
                        normalizer = StainNormalizationHE(target = target, stain_estimation_method = method)
                       
                        # apply on example image
                        im = normalizer.F(image)
                        # plot results
                        outPath = os.path.join(r"C:\Users\felix\Desktop\Neuro\AugmentOutput",file,method,target,testImg)
                        plt.imsave(outPath,im)
                        

    