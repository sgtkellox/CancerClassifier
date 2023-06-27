
import os
import numpy as np 

import cv2

import math

from filter_utils import *
from openslide import PROPERTY_NAME_COMMENT, PROPERTY_NAME_OBJECTIVE_POWER

from tile_utils import  calcPixelPosition

import argparse





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
        if ".ini" in img:
            continue
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
        if w == 0 or h == 0:
            print("Warning: the slide "+ slide +" has dims 0 , 0")
            continue
        w = math.floor(w/tileSize)*10
        h = math.floor(h/tileSize)*10
        resultImage = drawResultImage(resultMap,w,h)

        imgPath = os.path.join(outpath,slide+ ".jpg")
        cv2.imwrite(imgPath, resultImage)




if __name__ == '__main__':


    argParser = argparse.ArgumentParser()

    argParser.add_argument("-w", "--slides", help="The path to the folder containing the slides")
    argParser.add_argument("-t", "--tiles", help="The path to the folder containing the tiles")
    argParser.add_argument("-s", "--size", type=int,default=500, help="The size of the tiles in pixels")
    argParser.add_argument("-o", "--out", help="The path to where the resulting images go")

    args = argParser.parse_args()

    tilesPath = args.tiles
  
    slidesPath = args.slides

    size = args.size
  
    outPath = args.out
   



    tilingResultVisualisation(tilesPath, size,slidesPath , outPath)





