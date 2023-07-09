import os
import numpy as np
from tile_utils import calcPixelPosition

import cv2

import tifffile

from filter_utils import *



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

def makeTileMap(imgs,slideWidth, slideHeight):

    tileMap = [["empty"for row in range(int(slideHeight/500))] for col in range(int(slideWidth/500))]
  
    for img in imgs:   
        if ".ini" in img:
            continue
        
        x,y = calcPixelPosition(img)

        x = int(x)
        y = int(y)
        tileMap[x][y] = img
        
    return tileMap


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


def drawResult(tiles,tilePath,slideWidth, slideHeight):

    slideWidth = slideWidth - (slideWidth % 500)
    slideHeight = slideHeight - (slideHeight % 500)

    tileMap = makeTileMap(tiles,slideWidth, slideHeight)
    filler = np.zeros((slideHeight, 1, 3), np.uint8)
    for i in range(len(tileMap)):
        if tileMap[i][0] == "empty":
            row = np.zeros((500, 500, 3), np.uint8)
        for j in range(1,len(tileMap[0])):
            if not tileMap[i][j] == "empty":
                camPath = os.path.join(tilePath,tileMap[i][j])
                image = cv2.imread(camPath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                row = np.vstack((row,image)) 
            else:
               placeholder = np.zeros((500, 500, 3), np.uint8)
               row = np.vstack((row,placeholder))
        filler = np.hstack((filler,row))
     
    return filler


def stitchTilesToTiffs(path, tileSize, slidePath, outPath):
    wsis = sortTilesByWSI(path)

    for slide in wsis:
        w , h = getWsiDimensions(slide,slidePath)
        if w == 0 or h == 0:
           print("Warning: the slide "+ slide +" has dims 0 , 0")
           continue
        resultImg = drawResult(wsis[slide],path,w, h)
        stitchedImgPath = os.path.join(outPath,slide+".tiff")
        tifffile.imwrite(stitchedImgPath, resultImg,  photometric='rgb')


if __name__ == '__main__':

    #path = r"C:\Users\felix\Desktop\neuro\thTests\65_100\tiles"
    #slidePath = r"C:\Users\felix\Desktop\neuro\thTests\slidesTHTest"
    #outPath = r"C:\Users\felix\Desktop\neuro\tiffStitchingTest"
    #stitchTilesToTiffs(path, 500, slidePath, outPath)

    img = tifffile.imread(r'C:\Users\felix\Desktop\neuro\tiffStitchingTest\Inf-O2-N18-2962-K-Q2.tiff')
    print(img.shape)
   



        
       

