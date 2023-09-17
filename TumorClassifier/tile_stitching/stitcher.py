import os
from tqdm import tqdm

add_dll_dir = getattr(os, "add_dll_directory", None)
vipsbin = r"C:\AI\vips-dev-8.14\bin"
if callable(add_dll_dir):
    add_dll_dir(vipsbin)
    print("added dir")
else:
    os.environ["PATH"] = os.pathsep.join((vipsbin, os.environ["PATH"]))
    print("added path")


    

import sys
import random
import pyvips

# this makes a 8-bit, mono image of 100,000 x 100,000 pixels, each pixel zero






def calcPixelPosition(image):
    splitP1 = image.split("_")
    x = int(splitP1[1])
    y = int(splitP1[2].split(".")[0])
    return x , y

def findWidhHeight(path):
    minX = 0
    maxX = 0
    minY = 0
    maxY = 0
    for image in os.listdir(path):
   
        x,y = calcPixelPosition(image)

        minX = min(minX,x)
        maxX = max(maxX,x)
        minY = min(minY,y)
        maxY = max(maxY,y)

    width = maxX+500-minX
    height = maxY + 500 - minY

    return width, height



def stitchFolder(path,result):
    
    isEven = True

    for image in tqdm(os.listdir(path)):
        imagePath = os.path.join(path,image)

        tile = pyvips.Image.new_from_file(imagePath, access='sequential')

        if isEven == True:
            tile *= [1, 1.2, 1]
            isEven = False
        else:
            tile *= [1.2, 1, 1]
            isEven = True


        x,y = calcPixelPosition(image)

        result = result.insert(tile,x,y)
    return result

def main(tilePath,safePath):
    w, h  = findWidhHeight(tilePath)
    #print(w)
    #print(h)
    im = pyvips.Image.black(w,h,bands=3)

    im = stitchFolder(tilePath,im)

    im.write_to_file(safePath)




    
main(r"C:\Users\felix\Desktop\fixedKryoTest\stitcherTest",r"C:\Users\felix\Desktop\fixedKryoTest\stitcherRes\res.tif")

