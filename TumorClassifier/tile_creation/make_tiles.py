#Author: Areeba Patel, some minor changes to make it compatible with latest python version by Felix Keller

import sys
import numpy as np

from openslide import open_slide

import os

import cv2


def get_slidename(slidepath):
    
    slide_id = slidepath.split("\\")[-1].split(".")[0]
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




def make_tiles(slidepath,outPath,size):

    
    slide = open_slide(slidepath)

    

    w = int(slide.properties["openslide.level[0].width"])
    h = int(slide.properties["openslide.level[0].height"])
    
    
    # extract dimensions and print them
    #[w, h] = slide.dimensions
    print(f"Slide dimensions are {w}x{h}.", flush=True)
    print("Tiling...", flush=True)

    

    
  
    for x in range(0, w-size, size):
        for y in range(0, h-size, size):

            tile=slide.read_region(location=(x,y), level=0,size=(size,size))
            tileRGB = tile.convert('RGB')
           
            tileNP = np.array(tileRGB)

            #f not aboveThreshOld(tileNP):
                #ontinue

           #else:         
            if tileIsGlasBorder(tileNP):
                continue
            else:              
                if np.average(tileNP)<235:
                    tilename = get_tilename(slidepath, x, y)
                    tilename = os.path.join(outPath,tilename)
                    tileNP = cv2.cvtColor(tileNP, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(tilename, tileNP)
            
            
                  

    print("Done.", flush=True) 
    
    
    return 



     
 

