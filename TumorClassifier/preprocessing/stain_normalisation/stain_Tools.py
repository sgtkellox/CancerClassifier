import staintools
import os 
import cv2

import argparse


def normStainForFolder(imagePath,outPath):
    images = os.listdir(imagePath)

    target = staintools.read_image(r"C:\Users\felix\Desktop\isa\GBM-N16-2056-K_24500_16500.jpg")

    for image in images:
        print(image)

        imPath = os.path.join(imagePath, image)

        to_transform = staintools.read_image(imPath)

        # Standardize brightness (optional, can improve the tissue mask calculation)
        target = staintools.LuminosityStandardizer.standardize(target)
        to_transform = staintools.LuminosityStandardizer.standardize(to_transform)

        # Stain normalize
        normalizer = staintools.StainNormalizer(method='macenko')
        normalizer.fit(target)
        transformed = normalizer.transform(to_transform)

        safePath =  os.path.join(outPath,image)
        print(safePath)
        transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
        cv2.imwrite(safePath, transformed)

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    #argParser.add_argument("-i", "--input", help="The path to the folder containing the slides")
    #argParser.add_argument("-o", "--out", help="The path to the folder where u want the tiles")

    #args = argParser.parse_args()

    #pathIn = args.input
  
    #pathOut = args.out

    pathIn = r"C:\Users\felix\Desktop\isa\in"
    pathOut = r"C:\Users\felix\Desktop\isa\out"

    normStainForFolder(pathIn,pathOut)