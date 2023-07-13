import sys

import os 

import argparse

from make_tiles import make_tiles

from preprocessing_slides import process_tiles
from preprocessing_slides import mask4
from preprocessing_slides import SIZE

import matplotlib.pyplot as plt

def sliceSortedFolder(pathIn,pathOut,size):
     files = os.listdir(pathIn)
     for file in files:
         filePath = os.path.join(pathIn,file)
         make_tiles(filePath, outPath=pathOut, size = size)
                           
     return




if __name__ == '__main__': 
    
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--input", help="The path to the folder containing the slides")
    argParser.add_argument("-o", "--out", help="The path to the folder where u want the tiles")
    argParser.add_argument("-s", "--size",type=int, default=500, help="size of the tiles in pixel")

    

    args = argParser.parse_args()

    pathIn = args.input
  
    pathOut = args.out

    size = args.size

    
    sliceSortedFolder(pathIn,pathOut,size)