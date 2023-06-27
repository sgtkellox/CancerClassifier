
import os
import shutil
import random
import math
from sysconfig import get_path

from image_utils.slide_utils  import getPreparationFromFileName
from mage_utils.slide_utils import getDiagnosisFromFileName
from mage_utils.slide_utils import multipleNNumber

import matplotlib.pyplot as plt

import sys


   







def listAllFilesInSplit(path,outpath):
    outpath = os.path.join(outpath,"doc.txt")
    
    for prep in os.listdir(path):
        prepPath = os.path.join(path,prep)
        for part in os.listdir(prepPath):
            partPath = os.path.join(prepPath,part)
            for diag in os.listdir(partPath):
                diagPath = os.path.join(partPath,diag)           
                for slide in os.listdir(diagPath):
                    with open(outpath, 'a') as doc:
                        doc.write(slide)
                        doc.write('\n')

                    
        
        


                    
    







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



def splitSortedTiles(inPath,outPath,):

    wsisAstro = sortTilesByWSI(inAstro)
    wsisGBM = sortTilesByWSI(inGBM)
    wsisOligo = sortTilesByWSI(inOligo)
   

    for i in range(math.floor(len(wsisAstro)*0.85)):
        randomWsi = random.choice(list(wsisAstro.keys()))
        #print(randomWsi)
        selected = wsisAstro[randomWsi]
        for img in selected:
            imgPathSrc = os.path.join(inAstro,img)
            imgPathDest = os.path.join(outPath,"train","Astro",img)
           
            shutil.copyfile(imgPathSrc,imgPathDest)
        del wsisAstro[randomWsi]

    
    for i in range(math.floor(len(wsisGBM)*0.85)):
        randomWsi = random.choice(list(wsisGBM.keys()))
        selected = wsisGBM[randomWsi]
        for img in selected:
            imgPathSrc = os.path.join(inGBM,img)
            imgPathDest = os.path.join(outPath,"train","GBM",img)
           
            shutil.copyfile(imgPathSrc,imgPathDest)
        del wsisGBM[randomWsi]

    for i in range(math.floor(len(wsisOligo)*0.85)):
        randomWsi = random.choice(list(wsisOligo.keys()))
        selected = wsisOligo[randomWsi]
        for img in selected:
            imgPathSrc = os.path.join(inOligo,img)
            imgPathDest = os.path.join(outPath,"train","Oligo",img)
           
            shutil.copyfile(imgPathSrc,imgPathDest)
        del wsisOligo[randomWsi]



    for i in range(math.ceil(len(wsisAstro)*0.1)):
        randomWsi = random.choice(list(wsisAstro.keys()))
        selected = wsisAstro[randomWsi]
        for img in selected:
            imgPathSrc = os.path.join(inAstro,img)
            imgPathDest = os.path.join(outPath,"val","Astro",img)
           
            shutil.copyfile(imgPathSrc,imgPathDest)
        del wsisAstro[randomWsi]

    for i in range(math.ceil(len(wsisGBM)*0.1)):
        randomWsi = random.choice(list(wsisGBM.keys()))
        selected = wsisGBM[randomWsi]
        for img in selected:
            imgPathSrc = os.path.join(inGBM,img)
            imgPathDest = os.path.join(outPath,"val","GBM",img)
           
            shutil.copyfile(imgPathSrc,imgPathDest)
        del wsisGBM[randomWsi]

    for i in range(math.ceil(len(wsisOligo)*0.1)):
        randomWsi = random.choice(list(wsisOligo.keys()))
        selected = wsisOligo[randomWsi]
        for img in selected:
            imgPathSrc = os.path.join(inOligo,img)
            imgPathDest = os.path.join(outPath,"val","Oligo",img)
           
            shutil.copyfile(imgPathSrc,imgPathDest)
        del wsisOligo[randomWsi]

    for i in range(math.floor(len(wsisAstro)*0.05)):
        randomWsi = random.choice(list(wsisAstro.keys()))
        selected = wsisAstro[randomWsi]
        for img in selected:
            imgPathSrc = os.path.join(inAstro,img)
            imgPathDest = os.path.join(outPath,"test","Astro",img)
           
            shutil.copyfile(imgPathSrc,imgPathDest)
        del wsisAstro[randomWsi]

    for i in range(math.floor(len(wsisGBM)*0.05)):
        randomWsi = random.choice(list(wsisGBM.keys()))
        selected = wsisGBM[randomWsi]
        for img in selected:
            imgPathSrc = os.path.join(inGBM,img)
            imgPathDest = os.path.join(outPath,"test","GBM",img)
           
            shutil.copyfile(imgPathSrc,imgPathDest)
        del wsisGBM[randomWsi]

    for i in range(math.floor(len(wsisOligo)*0.05)):
        randomWsi = random.choice(list(wsisOligo.keys()))
        selected = wsisOligo[randomWsi]
        for img in selected:
            imgPathSrc = os.path.join(inOligo,img)
            imgPathDest = os.path.join(outPath,"test","Oligo",img)
           
            shutil.copyfile(imgPathSrc,imgPathDest)
        del wsisOligo[randomWsi]



   
    

def collectNNumbersFromTileSet(path):
    nNumbers = set()
    slides = os.listdir(path)
    for slide in slides:
        slideBaseName = slide.split("_")[0]
        
        #print(slideBaseName)
        nNumbers.add(slideBaseName)
    return nNumbers
        


def checkForSuccesfullTiling(slidePath,tilePath):
    nNumbersTileSet =  collectNNumbersFromTileSet(tilePath)
    for slide in os.listdir(slidePath):
        slideBaseName = slide.split(".")[0]

        if not slideBaseName in nNumbersTileSet:
            print("missing" + slideBaseName)


def extractNNumberFromSlide(slide):

    parts = slide.split("-")
    nNumber = parts[1] + "-" + parts[2]

    return nNumber
            
def findSlidesWithNNumberCounter(path):
    for slide in os.listdir(path):
        nNumber = extractNNumberFromSlide(slide)
        if not nNumber[-1].isdigit():
            os.remove(slide) 
            print(slide)



if __name__ == '__main__':
    src = r"E:\split"

    
    dest = r"C:\Users\felix\Desktop\dataSetInfo"

    

    listAllFilesInSplit(src,dest)