
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

def loadData(trainDir, valDir):
    train_directory = 'train'
    valid_directory = 'test'
    # Batch size
    bs = 32
    # Number of classes
    num_classes = 10
    # Load Data from folders
    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
        
    }
    # Size of Data, to be used for calculating Average Loss and Accuracy
    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])
    test_data_size = len(data['test'])
    # Create iterators for the Data loaded using DataLoader module
    train_data = DataLoader(data['train'], batch_size=bs, shuffle=True)
    valid_data = DataLoader(data['valid'], batch_size=bs, shuffle=True)
    test_data = DataLoader(data['test'], batch_size=bs, shuffle=True)
    # Print the train, validation and test set data sizes
    train_data_size, valid_data_size, test_data_size
    return

def loadData():
    train_dataset = torchvision.datasets.ImageFolder(root='train')
    valid_dataset = torchvision.datasets.ImageFolder(root='valid')



