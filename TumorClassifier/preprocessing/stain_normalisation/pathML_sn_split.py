import numpy as np
import matplotlib.pyplot as plt

import cv2

from pathml.core import HESlide
from pathml.preprocessing import StainNormalizationHE
import os

import argparse

from folder_structure_creation import makeFolderStructure

import multiprocessing

import shutil


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
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    print("writing image to" + outPath)
                    cv2.imwrite(outPath,im)

                except:
                    corrupted = True
                    break
                # plot results
               
            if corrupted:
                break
            

def stainNormSplit(inPath, outPath):
    
    makeFolderStructure(outPath)
    
    
    for prep in os.listdir(inPath):
        prepPath = os.path.join(inPath,prep)
        destPath = os.path.join(outPath,prep)
        for part in os.listdir(prepPath):
            partPath = os.path.join(prepPath,part)
            destPartPath = os.path.join(destPath,part)
            for diag in os.listdir(partPath):
                diagPath = os.path.join(partPath,diag)
                destPartPath = os.path.join(destPartPath,diag)
                    
                cpus = multiprocessing.cpu_count()-4
                images = os.listdir(pathIn)

                numImages = len(images)

                procs = []
                cpus = int(cpus)

                #createOutPath(pathOut)

                for i in range(1,cpus):
                    subSet = images[int((i-1)*numImages/cpus):int(i*numImages/cpus)]
                    print("started processing subset" +str(i)+ " of " + str(cpus))
                    tilingProc = multiprocessing.Process(target=normStainForFolder,args=(diagPath,destPartPath,subSet))
                    procs.append(tilingProc)
                    tilingProc.start()

                for process in procs:
                    process.join()
                    
                    
                    


    
    
    





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