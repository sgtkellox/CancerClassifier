#Author: Areeba Patel, some minor changes to make it compatible with latest python version by Felix Keller

import sys
import numpy as np

from openslide import open_slide

import os

import cv2


def get_slidename(slidepath):
    dirName , slideName  = os.path.split(slidepath)
    
    slide_id = slideName.split(".")[0]
    return slide_id



def get_tilename(slidepath, x, y):
    
    slide_id = get_slidename(slidepath)
    tilename = f"{slide_id}_{x}_{y}.jpg"
    return tilename


def tileIsGlasBorder(tile):
    number_of_black_pix  = np.sum(tile == (0,0,0))

    if number_of_black_pix > 100:
        return True
    else:
        return False

def aboveThreshOld(tile):
    h, w, _ = tile.shape
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    pixels = cv2.countNonZero(thresh)
    ratio = (pixels/(h * w)) * 100
   

    if ratio < 50:
        return False
    else:
        return True




def make_tiles(slidepath,outPath,size, level = 0):

    
    slide = open_slide(slidepath)

    #print(slide.level_downsamples)

    #slidePropertyString = "openslide.level[" +str(level)+ "].width"


    w0 = int(slide.properties["openslide.level[0].width"])
    h0 = int(slide.properties["openslide.level[0].height"])
      
    print(f"Slide w0 dimensions are {w0}x{h0}.", flush=True)
    print("Tiling...", flush=True)
    print("-----")
    
    
    if level == 0:
        factor = 1
    elif level == 1:
        factor = 4
    elif level == 2:
        factor = 8
        
    grow = factor*  size

    for x in range(0, w0-grow, grow):
        for y in range(0, h0-grow, grow):

            tile=slide.read_region(location=(x,y), level=level,size=(size,size))
            tileRGB = tile.convert('RGB')
           
            tileNP = np.array(tileRGB)

            #f not aboveThreshOld(tileNP):
                #ontinue

           #else:         
            if tileIsGlasBorder(tileNP):
                continue
            else:              
                if np.average(tileNP)<235:
                    tilename = get_tilename(slidepath, int(x/factor), int(y/factor))
                    tilename = os.path.join(outPath,tilename)
                    tileNP = cv2.cvtColor(tileNP, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(tilename, tileNP)
            
            
              
    print("Done.", flush=True) 
    
    
    return 



     
 

