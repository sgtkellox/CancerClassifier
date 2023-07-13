#Author: Areeba Patel, some minor changes to make it compatible with latest python version by Felix Keller

import sys
import numpy as np
import pickle
from openslide import open_slide
import openslide

import os

import cv2


def get_slidename(slidepath):
    
    slide_id = slidepath.split("\\")[-1].split(".")[0]
    return slide_id



def get_tilename(slidepath, x, y):
    
    slide_id = get_slidename(slidepath)
    tilename = f"{slide_id}_{x}_{y}.jpg"
    return tilename



def make_tiles(slidepath,outPath,size):

    print(slidepath)
    
    slide = open_slide(slidepath)
    
    # extract dimensions and print them
    [w, h] = slide.dimensions
    print(f"Slide dimensions are {w}x{h}.", flush=True)
    print("Tiling...", flush=True)
  
    for x in range(0, w-size, size):
        for y in range(0, h-size, size):

            tile=slide.read_region(location=(x,y), level=0, size=(size,size))
            tileRGB = tile.convert('RGB')
            tileNP = np.array(tileRGB)
            if np.average(tileNP)<235:
                tilename = get_tilename(slidepath, x, y)
                tilename = os.path.join(outPath,tilename)
                tileNP = cv2.cvtColor(tileNP, cv2.COLOR_BGR2RGB)
                cv2.imwrite(tilename, tileNP)
            
            
                  

    print("Done.", flush=True) 
    
    
    return 



     
 

