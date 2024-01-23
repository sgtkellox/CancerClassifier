


import matplotlib.pyplot as plt
import numpy as np

import cv2

import argparse

import os





def getSlideName(slide):
    slideName = slide.split("\\")[-1]
    slideName = slideName.split(".")[0]
    return slideName
    


def processFile(file, pathOut):
    
    
    print(file)
    
    image = cv2.imread(file) #bild laden
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR to RGB 
    
    #red = image[:, :, 0]
    #green = image[:, :, 1]
    blue = image[:, :, 2]
    
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    
    
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
    resizedOridPath = os.path.join(pathOut,getSlideName(file)+"_resizedOrig.png")
    
    cv2.imwrite(resizedOridPath,imageResized)
    
    ret, thresh1 = cv2.threshold(grey, 80, 255, cv2.THRESH_BINARY) 


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))

    #opening_img = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE , kernel )
    safePath = os.path.join(pathOut,getSlideName(file)+".jpg")
    
    resized = cv2.resize(thresh1, (500, 500))
    
    resizedPath = os.path.join(pathOut,getSlideName(file)+"_resized.png")
    
    cv2.imwrite(resizedPath,resized)
    cv2.imwrite(safePath,thresh1)
    

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
    


def processFolder(pathIn, pathOut):
    for file in os.listdir(pathIn):
        if not file.endswith(".tif"):
            continue
        filePath = os.path.join(pathIn,file)
        processFile(filePath, pathOut)
        

def processAllFolders(pathIn, pathOut): 
     for folder in os.listdir(pathIn):
        print("------------------------------")
        folderpath = os.path.join(pathIn,folder)
        safePath = os.path.join(pathOut,folder)
        if not os.path.isdir(safePath):
            os.mkdir(safePath)
        if os.path.isfile(folderpath):
            continue
        processFolder(folderpath,safePath)
        

if __name__ == '__main__':
    pathIn = r"F:\coresTiff"
    pathOut = r"F:\maskedCores2"
    
    processAllFolders(pathIn,pathOut)
    
    
        
        
        
        
        

