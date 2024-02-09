


import matplotlib.pyplot as plt
import numpy as np

import cv2

import argparse

import os

from remove_cores import removeBadCores
from remove_empty import thresholdGrey

from split_dataset import split





def getSlideName(slide):
    slideName = slide.split("\\")[-1]
    slideName = slideName.split(".")[0]
    return slideName


def centerCrop(file,outpath):
    image = cv2.imread(file)
    width = image.shape[1]
    height =  image.shape[0]
    
    crop = image[int(height*0.2):int(height*0.8),int(width*0.2):int(width*0.8)]
    return crop
    
    

def processFile(file, pathOut, pathResized):
    
    
    print(file)
    
    image = cv2.imread(file) #bild laden
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR to RGB 
    
    #red = image[:, :, 0]
    #green = image[:, :, 1]
    blue = image[:, :, 2]
    
    #grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    
    
    #redPath = os.path.join(pathOut,getSlideName(file)+"_red.png")
    #cv2.imwrite(redPath,red)
    
    #greenPath = os.path.join(pathOut,getSlideName(file)+"_green.png")
    #cv2.imwrite(greenPath,green)
    
    #bluePath = os.path.join(pathOut,getSlideName(file)+"_blue.png")
    #cv2.imwrite(bluePath,blue)
    
    #greyPath = os.path.join(pathOut,getSlideName(file)+"_grey.png")
    #cv2.imwrite(greyPath,grey)
    
    
    imageResized = cv2.resize(image, (500, 500))
    imageResized = cv2.cvtColor(imageResized, cv2.COLOR_BGR2RGB)
    #resizedOridPath = os.path.join(pathOut,getSlideName(file)+".png")
    
    cv2.imwrite(pathOut,imageResized)
    
    
    ret, thresh1 = cv2.threshold(blue, 100, 255, cv2.THRESH_BINARY) 


    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))

    #opening_img = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE , kernel )
    #safePath = os.path.join(pathOut,getSlideName(file)+".jpg")
    
    resized = cv2.resize(thresh1, (500, 500))
    
    #resizedPath = os.path.join(pathOut,getSlideName(file)+"_resized.png")
    
    cv2.imwrite(pathResized,resized)
    #cv2.imwrite(safePath,thresh1)
    
    
    

def testMask(file, pathOut):
    print(file)
    
    
    image = cv2.imread(file)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blue = image[:, :, 2]
    
    ret, thresh1 = cv2.threshold(blue, 80, 255, cv2.THRESH_BINARY) 


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))

    mask = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE , kernel )
    safePath = os.path.join(pathOut,getSlideName(file)+".jpg")

    print("image shape " + str(image.shape))
    print("mask shape " + str(mask.shape))
    
    maskPath = os.path.join(pathOut,getSlideName(file)+"_mask.jpg")
    
    result = image.copy()
    result[mask == 0] = 0
    result[mask != 0] = image[mask != 0]
    
    resized = cv2.resize(mask, (500, 500))
    cv2.imwrite(maskPath,resized)
    
    cv2.imwrite(safePath,result)
    
def mixImages(pathIn, pathOut):
    
    bwPath = os.path.join(pathOut,"grey")
    rgbPathOut = os.path.join(pathOut,"rgb")
    
    os.mkdir(bwPath)
    os.mkdir(rgbPathOut)
   
    for folder in os.listdir(pathIn):
        print("looking up" + folder)
        identifier = folder.split("-")[2]        
        folderPath = os.path.join(pathIn,folder)
        for file in os.listdir(folderPath):
            if not file.endswith(".tif"):
                continue
            else:
                newFileName = os.path.join(rgbPathOut, str(identifier)+"-"+getSlideName(file)+".jpg")
                filePath = os.path.join(folderPath,file)
                greyPath = os.path.join(bwPath,str(identifier)+"-"+getSlideName(file)+".jpg")
                print(greyPath)
                processFile(filePath,newFileName,greyPath)
    


def processFolder(pathIn, pathOut):
    for file in os.listdir(pathIn):
        if not file.endswith(".tif"):
            continue
        filePath = os.path.join(pathIn,file)
        outPath = os.path.join(pathOut,file)
        
        crop = centerCrop(filePath,outPath)
        
        processFile(filePath, pathOut)
        

def processAllFolders(pathIn, pathOut): 
     for folder in os.listdir(pathIn):
        print("------------------------------")
        folderpath = os.path.join(pathIn,folder)
        safePath = os.path.join(pathOut,folder)
        removeBadCores(pathIn)
        if not os.path.isdir(safePath):
            os.mkdir(safePath)
        if os.path.isfile(folderpath):
            continue
        processFolder(folderpath,safePath)
        

if __name__ == '__main__':
    pathIn = r"D:\crop"
    pathOut = r"D:\mix"
    badCores = r"C:\Users\felix\Downloads\not_representative_cores.txt"
    binPath = r"D:\bin"
    labelFile = r"D:\label.txt"
    splitPath = r"D:\split"
    removeBadCores(badCores,pathIn,binPath)
    mixImages(pathIn, pathOut)
    greyPath = os.path.join(pathOut,"grey")
    thresholdGrey(greyPath, 254.8)
    split(greyPath,labelFile,splitPath)
    
    
    
    
    
        
        
        
        
        

