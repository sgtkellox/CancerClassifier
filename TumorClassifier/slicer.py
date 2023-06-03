import os
import shutil
import random
import math
from sysconfig import get_path

import sys

from preprocessing_slides import process_tiles
from preprocessing_slides import mask4
from preprocessing_slides import SIZE

import matplotlib.pyplot as plt

def sliceKryos(pathIn,pathOut):
     files = os.listdir(pathIn)
     for file in files:
            if file.startswith("GBM"):           
                if "K" in file:
                    #dst = os.path.join(GBMfolder,"Kryo",file)
                    filePath = os.path.join(pathIn,file)
                    process_tiles(filePath,mask4, outPath=pathOut)
            elif file.startswith("O"):               
                if  "K" in file:
                    filePath = os.path.join(pathIn,file)
                    process_tiles(filePath,mask4, outPath=pathOut)

            elif file.startswith("A"):              
                if "K" in file:
                    filePath = os.path.join(pathIn,file)
                    process_tiles(filePath,mask4, outPath=pathOut)
     return



if __name__ == '__main__':
    pathIn = sys.argv[1]
    pathOut = sys.argv[2]


    sliceKryos(pathIn,pathOut)



