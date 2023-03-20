
import os
import shutil
import random
import math
from sysconfig import get_path

from preprocessing_slides import process_tiles
from preprocessing_slides import mask4
from preprocessing_slides import SIZE

import matplotlib.pyplot as plt



project_name='BrainCancerClassifier'

GBMfolder = ""
oligoFolder = ""
astroFolder = ""







def sortImagesByFileName(path):
    files = os.listdir(path)
    for file in files:
        if file.startswith("GBM"):
            if "T" in file:
                #filePath = os.path.join(, file)
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"E:\ClassifierImages\GBM\Touch")
            elif "Q" in file:
                #dst = os.path.join(GBMfolder,"Smear",file)
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"E:\ClassifierImages\GBM\Smear")
            elif "K" in file:
                #dst = os.path.join(GBMfolder,"Kryo",file)
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"E:\ClassifierImages\GBM\Kryo")
        elif file.startswith("O"):
            if "T" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Oligo\Touch")
            elif "Q" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Oligo\Smear")
            elif "K" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Oligo\Kryo")

        elif file.startswith("A"):
            if "T" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Astro\Touch")
            elif "Q" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Astro\Smear")
            elif "K" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Astro\Kryo")
    return


def sortLeftOverSlides(path):
    files = os.listdir(path)
    for file in files:
        if file.startswith("GBM"):
            if "T" in file:
                #filePath = os.path.join(, file)
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"E:\ClassifierImages\GBM\Touch")
            elif "Q" in file:
                #dst = os.path.join(GBMfolder,"Smear",file)
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\testSetsSmear\GBM")
            elif "K" in file:
                #dst = os.path.join(GBMfolder,"Kryo",file)
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"E:\ClassifierImages\GBM\Kryo")
        elif file.startswith("O"):
            if "T" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Oligo\Touch")
            elif "Q" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\testSetsSmear\Oligo")
            elif "K" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Oligo\Kryo")

        elif file.startswith("A"):
            if "T" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Astro\Touch")
            elif "Q" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\testSetsSmear\Astro")
            elif "K" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Astro\Kryo")
    return


def makeTrainValTestSplit(path,trainPath,valPath,testPath):
    files = os.listdir(path)
    trainSize = math.floor(0.7*len(files))
    valSize = math.floor(0.15 * len(files))
    
    for i in range(trainSize):
        chosenFile = random.choice(files)
        shutil.copyfile(os.path.join(path,chosenFile),os.path.join(trainPath,chosenFile))
        files.remove(chosenFile)
    for j in range(valSize):
        chosenFile = random.choice(files)
        shutil.copyfile(os.path.join(path,chosenFile),os.path.join(valPath,chosenFile))
        files.remove(chosenFile)
    for file in files:
        shutil.copyfile(os.path.join(path,file),os.path.join(testPath,file))
    return

def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))




sortLeftOverSlides(r"E:\KryoForTest")
