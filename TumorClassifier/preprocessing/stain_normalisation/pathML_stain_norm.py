import numpy as np
import matplotlib.pyplot as plt

import cv2

from pathml.core import HESlide
from pathml.preprocessing import StainNormalizationHE
import os

import argparse

def createOutPath(path):
    mPath = os.path.join(path,"macenko")
    if not os.path.isdir(mPath):
        os.mkdir(mPath)
    vPath = os.path.join(path,"vahadane")
    if not os.path.isdir(vPath):
        os.mkdir(vPath)

    toCreate = os.path.join(mPath,"normalize")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)

    toCreate = os.path.join(mPath,"hematoxylin")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)

    toCreate = os.path.join(mPath,"eosin")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)

    toCreate = os.path.join(vPath,"normalize")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)

    toCreate = os.path.join(vPath,"hematoxylin")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)

    toCreate = os.path.join(vPath,"eosin")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)


def normStainForFolder(pathIn,pathOut):
    
    createOutPath(pathOut)
    
    copiedFiles = os.listdir(os.path.join(pathOut,"macenko","hematoxylin"))
        
            
    testImgs = os.listdir(pathIn)
    testImgs.sort()

    print(testImgs[0:50])
            
    for testImg in testImgs:
        if not testImg.endswith(".jpg"):
            continue
        if testImg in copiedFiles:
            print("file " + testImg+ " allready copied")
            continue
        imgPath = os.path.join(pathIn,testImg)
        image = cv2.imread(imgPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i, method in enumerate(["macenko", "vahadane"]):
            for j, target in enumerate(["normalize", "hematoxylin", "eosin"]):
                # initialize stain normalization object
                normalizer = StainNormalizationHE(target = target, stain_estimation_method = method)
                       
                # apply on example image
                im = normalizer.F(image)
                # plot results
                outPath = os.path.join(pathOut,method,target,testImg)
                plt.imsave(outPath,im)









if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--input", help="The path to the folder containing the slides")
    argParser.add_argument("-o", "--out", help="The path to the folder where u want the tiles")

    args = argParser.parse_args()

    pathIn = args.input
  
    pathOut = args.out

    normStainForFolder(pathIn,pathOut)
     
                        

    