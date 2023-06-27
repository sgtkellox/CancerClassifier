import os
import shutil
import random
import math
import sys

sys.path.insert(0, r'..\image_utils\slide_utils')

from image_utils.slide_utils  import getPreparationFromFileName
from image_utils.slide_utils import getDiagnosisFromFileName
from image_utils.slide_utils import multipleNNumber

import matplotlib.pyplot as plt

import argparse



def sortByPreparation(files):
    smearSlides = []
    kryoSlides = []
    touchSlides = []
    
    for file in files:
        if getPreparationFromFileName(file) == "Q":
            smearSlides.append(file)
        elif getPreparationFromFileName(file) == "K":
            kryoSlides.append(file)
        elif getPreparationFromFileName(file) == "T":
            touchSlides.append(file)
     
    return kryoSlides, smearSlides, touchSlides


def makeSplit(slides):

    trainSize = math.floor(0.8*len(slides))
    valSize = math.floor(0.15*len(slides))
    testSize = math.floor(0.05*len(slides))

    trainSet = []
    valSet = []
    testSet = []

    train = 0
    val = 0
    test = 0

    while test< testSize:
        
        chosenFile = random.choice(slides)
        if not multipleNNumber(chosenFile):
            testSet.append(chosenFile)
            slides.remove(chosenFile)
            test+=1
        else:
            trainSet.append(chosenFile)
            slides.remove(chosenFile)
            train+=1

    while val< valSize:
        
        chosenFile = random.choice(slides)
        if not multipleNNumber(chosenFile):
            valSet.append(chosenFile)
            slides.remove(chosenFile)
            val+=1
        else:
            trainSet.append(chosenFile)
            slides.remove(chosenFile)
            train+=1
    trainSet.extend(slides)

    return trainSet, valSet, testSet


def sortByDiagnosis(splitPart,path,srcPath):
    

    astroPath = os.path.join(path,"Astro")
    gbmPath = os.path.join(path,"GBM")
    oligoPath = os.path.join(path,"Oligo")

    for slide in splitPart:
        if getDiagnosisFromFileName(slide) == "A":
            print( "moving " + slide + "  to " + os.path.join(astroPath,slide))
            shutil.move(os.path.join(srcPath,slide),os.path.join(astroPath,slide))
        elif getDiagnosisFromFileName(slide) =="GBM":
            shutil.move(os.path.join(srcPath,slide),os.path.join(gbmPath,slide))
        elif getDiagnosisFromFileName(slide) =="O":
            shutil.move(os.path.join(srcPath,slide),os.path.join(oligoPath,slide))


def sortAndSplit(inPath,outPath):
    slides = os.listdir(inPath)

    kryoSet , smearSet, touchSet = sortByPreparation(slides)

    kryoPath = os.path.join(outPath,"kryo")
    smearPath = os.path.join(outPath,"smear")
    touchPath = os.path.join(outPath,"touch")

    kryoTrain , kryoVal, kryoTest = makeSplit(kryoSet)


    kryoTrainPath = os.path.join(kryoPath,"train")
    kryoValPath = os.path.join(kryoPath,"val")
    kryoTestPath = os.path.join(kryoPath,"test")

    sortByDiagnosis(kryoTrain,kryoTrainPath,inPath)
    sortByDiagnosis(kryoVal,kryoValPath,inPath)
    sortByDiagnosis(kryoTest,kryoTestPath,inPath)

    smearTrain, smearVal, smearTest = makeSplit(smearSet)

    smearTrainPath = os.path.join(smearPath,"train")
    smearValPath = os.path.join(smearPath,"val")
    smearTestPath = os.path.join(smearPath,"test")

    sortByDiagnosis(smearTrain,smearTrainPath,inPath)
    sortByDiagnosis(smearVal,smearValPath,inPath)
    sortByDiagnosis(smearTest,smearTestPath,inPath)

    touchTrain, touchVal, touchTest = makeSplit(touchSet)

    touchTrainPath = os.path.join(touchPath,"train")
    touchValPath = os.path.join(touchPath,"val")
    touchTestPath = os.path.join(touchPath,"test")

    sortByDiagnosis(touchTrain,touchTrainPath,inPath)
    sortByDiagnosis(touchVal,touchValPath,inPath)
    sortByDiagnosis(touchTest,touchTestPath,inPath)


def split(inPath,outPath):
   
    kryoIn = os.path.join(inPath,"kryo")
    smearIn = os.path.join(inPath,"smear")
    touchIn = os.path.join(inPath,"touch")

   

    kryoSet = os.listdir(kryoIn)
    smearSet = os.listdir(smearIn)
    touchSet = os.listdir(touchIn)

    kryoPath = os.path.join(outPath,"kryo")
    smearPath = os.path.join(outPath,"smear")
    touchPath = os.path.join(outPath,"touch")

    kryoTrain , kryoVal, kryoTest = makeSplit(kryoSet)


    kryoTrainPath = os.path.join(kryoPath,"train")
    kryoValPath = os.path.join(kryoPath,"val")
    kryoTestPath = os.path.join(kryoPath,"test")

    sortByDiagnosis(kryoTrain,kryoTrainPath,kryoIn)
    sortByDiagnosis(kryoVal,kryoValPath,kryoIn)
    sortByDiagnosis(kryoTest,kryoTestPath,kryoIn)

    smearTrain, smearVal, smearTest = makeSplit(smearSet)

    smearTrainPath = os.path.join(smearPath,"train")
    smearValPath = os.path.join(smearPath,"val")
    smearTestPath = os.path.join(smearPath,"test")

    sortByDiagnosis(smearTrain,smearTrainPath,smearIn)
    sortByDiagnosis(smearVal,smearValPath,smearIn)
    sortByDiagnosis(smearTest,smearTestPath,smearIn)

    touchTrain, touchVal, touchTest = makeSplit(touchSet)

    touchTrainPath = os.path.join(touchPath,"train")
    touchValPath = os.path.join(touchPath,"val")
    touchTestPath = os.path.join(touchPath,"test")

    sortByDiagnosis(touchTrain,touchTrainPath,touchIn)
    sortByDiagnosis(touchVal,touchValPath,touchIn)
    sortByDiagnosis(touchTest,touchTestPath,touchIn)


if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-s", "--slides", help="The path to the folder containing the slides")
    argParser.add_argument("-t", "--tiles", help="The path to the folder containing the split structure of the tiles")
    
    args = argParser.parse_args()

    inPath = args.slides
    outPath = args.tiles

    split(inPath,outPath)



    




