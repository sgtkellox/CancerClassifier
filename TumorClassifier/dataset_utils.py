
import os
import shutil
import random
import math
from sysconfig import get_path
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
#from preprocessing_slides import process_tiles
#from preprocessing_slides import mask4
#from preprocessing_slides import SIZE

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
                process_tiles(filePath,mask4, outPath=r"E:\ClassifierImages\Oligo\Touch")
            elif "Q" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"E:\ClassifierImages\Oligo\Smear")
            elif "K" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"E:\ClassifierImages\Oligo\Kryo")

        elif file.startswith("A"):
            if "T" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"E:\ClassifierImages\Astro\Touch")
            elif "Q" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"E:\ClassifierImages\Astro\Smear")
            elif "K" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"E:\ClassifierImages\Astro\Kryo")
    return


def makeTrainValTestSplit(path,trainPath,valPath,testPath):
    files = os.listdir(path)
    trainSize = 11900
    valSize = 2550
    
    for i in range(trainSize):
        chosenFile = random.choice(files)
        if chosenFile.endswith(".jpg"):
            src = os.path.join(path,chosenFile)
            dst = os.path.join(trainPath,chosenFile)
            shutil.copy(src,dst)
            files.remove(chosenFile)
    for j in range(valSize):
        chosenFile = random.choice(files)
        if chosenFile.endswith(".jpg"):
            shutil.copy(os.path.join(path,chosenFile),os.path.join(valPath,chosenFile))
            files.remove(chosenFile)
    for k in range(valSize):
        chosenFile = random.choice(files)
        if chosenFile.endswith(".jpg"):
            shutil.copy(os.path.join(path,chosenFile),os.path.join(testPath,chosenFile))
    return

def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))


def makeSampleDistributionHistogramm(rootPath):
    trainPath = os.path.join(rootPath,"train")
    result = {}
    classes = os.listdir(trainPath)
    for label in classes:
        if label.is_dir():
            currentClassPath = os.path.join(trainPath,label)
            files = os.listdir(currentClassPath)
            entry = {"train": len(files), "val": 0, "test":0}
            result[label] = entry
    valPath = os.path.join(rootPath,"val")
    classes = os.listdir(valPath)
    for label in classes:
        if label.is_dir():
            currentClassPath = os.path.join(trainPath,label)
            files = os.listdir(currentClassPath)
            result[label]["val"] = len(files)
    testPath = os.path.join(rootPath,"test")
    classes = os.listdir(testPath)
    for label in classes:
        if label.is_dir():
            currentClassPath = os.path.join(testPath,label)
            files = os.listdir(currentClassPath)
            result[label]["test"] = len(files)
           
    return result


makeTrainValTestSplit(r"E:\ClassifierImages\Oligo\Smear",r"E:\ClassifierSplit\train\Oligo",r"E:\ClassifierSplit\val\Oligo",r"E:\ClassifierSplit\test\Oligo")