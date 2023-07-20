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
         if file.endswith(".svs"):
            filePath = os.path.join(pathIn,file)
            make_tiles(filePath, outPath=pathOut, size = size)
                           
     return




if __name__ == '__main__': 
    
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--input", help="The path to the folder containing the slides")
    argParser.add_argument("-o", "--out", help="The path to the folder where u want the tiles")

    args = argParser.parse_args()

    pathIn = args.input
  
    pathOut = args.out
    
    sliceSortedFolder(pathIn,pathOut,500)