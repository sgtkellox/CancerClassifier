import os
import shutil

import argparse

import cv2







if __name__ == '__main__':
    argParser = argparse.ArgumentParser()

    
    argParser.add_argument("-p", "--path", help="Folder containing the files to be counted")

    args = argParser.parse_args()

    path = args.path

    pathIn = r""

    pathOut = r""

    


    for imageName in os.listdir(pathIn):
        imgPath = os.path.join(pathIn,imageName)


   