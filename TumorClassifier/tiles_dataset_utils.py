
import os
import numpy as np 

import cv2

import math

from filter_utils import *
from openslide import PROPERTY_NAME_COMMENT, PROPERTY_NAME_OBJECTIVE_POWER

from wsi_utils import extractIndetifierFromSlide, calcPixelPosition





def sortTilesByWSI(path):

    wsis = {}

    for img in os.listdir(path):

        wsiName = img.split("_")[0]

        if wsiName in wsis:
            wsis[wsiName].append(img)
        else:
            wsis[wsiName] = []
            wsis[wsiName].append(img)
    return wsis

def calcSlideResultWithPositions(imgs):

    result = []
    for img in imgs:   
        resultEntry = 1
        x,y = calcPixelPosition(img)

        resEntry = [resultEntry,y,x]

        result.append(resEntry)

    return result

def drawResultImage(resultsArray, wsiWidth, wsiHeight):
     result = np.zeros((wsiHeight, wsiWidth, 3), np.uint8)
     result.fill(255)

     for entry in resultsArray:
            if entry[0]==1:
                result[int(entry[1])*10:int(entry[1])*10+10,int(entry[2])*10:int(entry[2])*10+10] = [0, 0, 0]
            
     return result


def getWsiDimensions(nNumber, slidePath):
    slides = os.listdir(slidePath)
   
    for wsi in slides:
        wsiNnumber = wsi.split(".")[0]
        
        if wsiNnumber == nNumber:
               
            slidePath = os.path.join(slidePath,wsi)
            slide = open_slide(slidePath)
            a = slide.dimensions
            return a[0] , a[1]
                    
    return 0, 0
    

def tilingResultVisualisation(path, tileSize, slidePath, outpath):
    
    wsis = sortTilesByWSI(path)

    for slide in wsis:
        resultMap = calcSlideResultWithPositions(wsis[slide])
        w , h = getWsiDimensions(slide,slidePath)
        w = math.floor(w/tileSize)*10
        h = math.floor(h/tileSize)*10
        resultImage = drawResultImage(resultMap,w,h)

        imgPath = os.path.join(outpath,slide+ ".jpg")
        print(imgPath)
        cv2.imwrite(imgPath, resultImage)




if __name__ == '__main__':

    split = "train"
    diag = "Oligo"

    tilebasePath = r"C:\Users\felix\Desktop\neuro\kryo"
    tilePath = os.path.join(tilebasePath, split, diag)

    splitBasePath = r"E:\split\kryo"
    splitPath = os.path.join(splitBasePath, split, diag)

    outBasePath = r"C:\Users\felix\Desktop\neuro\kryoResImgs"
    outPath = os.path.join(outBasePath, split, diag)



    tilingResultVisualisation(tilePath, 500,splitPath , outPath)





