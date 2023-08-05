import numpy as np
import matplotlib.pyplot as plt

import cv2

from pathml.core import HESlide
from pathml.preprocessing import StainNormalizationHE
import os

import argparse

import multiprocessing

def createOutPath(path):
    mPath = os.path.join(path,"macenko")
    if not os.path.isdir(mPath):
        os.mkdir(mPath)
   

    toCreate = os.path.join(mPath,"normalize")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)

    toCreate = os.path.join(mPath,"hematoxylin")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)

    toCreate = os.path.join(mPath,"eosin")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)

    


def normStainForFolder(pathIn,pathOut, images):
    
    
    
    copiedFiles = os.listdir(pathOut)
        
            
            
    for imageName in images:
        if not imageName.endswith(".jpg"):
            continue
        if imageName in copiedFiles:
            print("file " + imageName+ " allready copied")
            continue
        imgPath = os.path.join(pathIn,imageName)
        image = cv2.imread(imgPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        corrupted = False
        for i, method in enumerate(["macenko"]):
            for j, target in enumerate(["normalize"]):
                # initialize stain normalization object
                normalizer = StainNormalizationHE(target = target, stain_estimation_method = method)
                       
                # apply on example image
                try:
                    im = normalizer.F(image)
                    outPath = os.path.join(pathOut,imageName)
                    print("writing image to" + outPath)
                    cv2.imwrite(outPath,im)

                except:
                    corrupted = True
                    break
                # plot results
               
            if corrupted:
                break





if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--input", help="The path to the folder containing the slides")
    argParser.add_argument("-o", "--out", help="The path to the folder where u want the tiles")

    args = argParser.parse_args()

    pathIn = args.input
  
    pathOut = args.out

    cpus = multiprocessing.cpu_count()-4

    print("Number of cpu : ", cpus)

    images = os.listdir(pathIn)

    numImages = len(images)

    procs = []
    cpus = int(cpus)

    #createOutPath(pathOut)

    for i in range(1,cpus):
        subSet = images[int((i-1)*numImages/cpus):int(i*numImages/cpus)]
        print("started processing subset" +str(i)+ " of " + str(cpus))
        tilingProc = multiprocessing.Process(target=normStainForFolder,args=(pathIn,pathOut,subSet))
        procs.append(tilingProc)
        tilingProc.start()

    for process in procs:
        process.join()



        
     
                        

    