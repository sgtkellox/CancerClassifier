
from tqdm import tqdm

import numpy as np

import os
add_dll_dir = getattr(os, "add_dll_directory", None)
vipsbin = r"C:\AI\vips-dev-8.14\bin"
if callable(add_dll_dir):
    add_dll_dir(vipsbin)
    print("added dir")
else:
    os.environ["PATH"] = os.pathsep.join((vipsbin, os.environ["PATH"]))
    print("added path")

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


    

import sys
import random
import pyvips

# this makes a 8-bit, mono image of 100,000 x 100,000 pixels, each pixel zero

# map vips formats to np dtypes






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

        

        mem_img = tile.write_to_memory()

        np_3d = np.ndarray(buffer=mem_img,dtype=format_to_dtype[tile.format],shape=[tile.height, tile.width, tile.bands])

        #greenOverLay = 


        if isEven == True:
            np_3d[:,:,0] = 255
            isEven = False
        else:
            np_3d[:,:,1] = 255
            isEven = True

        height, width, bands = np_3d.shape
        linear = np_3d.reshape(width * height * bands)
        vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                          dtype_to_format[str(np_3d.dtype)])


        x,y = calcPixelPosition(image)

        result = result.insert(vi,x,y)
    return result

def main(tilePath,safePath):
    w, h  = findWidhHeight(tilePath)
    #print(w)
    #print(h)
    im = pyvips.Image.black(w,h,bands=3)

    im = stitchFolder(tilePath,im)


    #im.tiffsave(safePath, tile=True, compression='lzw', bigtiff=True, pyramid=True, Q=80)

    im.tiffsave(safePath, compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
                 tile=True, tile_width=512, tile_height=512, #rgbjpeg=True,
                 pyramid=True,  bigtiff=True)

    #im.write_to_file(safePath)



    
main(r"C:\Users\felix\Desktop\neuro\stitcherTest",r"C:\Users\felix\Desktop\neuro\stitcherRes\res.tif")





