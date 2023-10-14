
import os
import argparse
import math
import random
from folder_structure_creation import makeFolderStructure
from image_utils.tile_utils import getPreparation, getDiagnosis
import shutil


def sortTilesByWSI(path):

    wsis = {}
    for img in os.listdir(path):

        wsiName = img.split("_")[0]

        if wsiName in wsis:
            wsis[wsiName].append(img)
        else:
            wsis[wsiName] = []
            wsis[wsiName].append(img)

    print(wsis)
    return wsis


def validateInput(trainRatio, valRatio, testRation):

    if trainRatio + valRatio + testRation == 1:
        return True
    else: 
        return False

def suitedForVal(wsi):
    identifier = wsi.split("_")[0]
    num = identifier.split("-")[3]
    if num[-1].isdigit():
        return False
    else: 
        return True





def split(wsis, trainRatio, valRatio, testRatio):

    setSize = len(wsis)
  

    valSize = math.floor(setSize*valRatio)

    testSize = math.floor(setSize*testRatio)
    
    trainSet = []
    valSet = []
    testSet = []

    x = 0
    while x<valSize:
        randomWsi ,value = random.choice(list(wsis.items()))
        #print(randomWsi)
        if suitedForVal(randomWsi):
            valSet = valSet + wsis[randomWsi]
            del wsis[randomWsi]
            x +=1

    x = 0
    while x < testSize:
        randomWsi ,value = random.choice(list(wsis.items()))
        if suitedForVal(randomWsi):
            testSet = testSet + wsis[randomWsi]
            del wsis[randomWsi]
            x +=1
    
            
    for wsi in wsis:
        trainSet = trainSet + wsis[wsi]

    return trainSet , valSet, testSet

def matchListLength(longList, shortList):
    diff = len(longList)-len(shortList)
    diffList = ["---"]*diff

    shortList = shortList + diffList
    return shortList

    
def printSplit( trainSet, valSet , testSet):

    trainWsis = []
    valWsis = []
    testWsis = []

    for image in trainSet:
         wsiName = image.split("_")[0]
         if not wsiName in trainWsis:
             trainWsis.append(wsiName)
    for image in valSet:
         wsiName = image.split("_")[0]
         if not wsiName in valWsis:
             valWsis.append(wsiName)
    for image in testSet:
         wsiName = image.split("_")[0]
         if not wsiName in testWsis:
             testWsis.append(wsiName)


    print(" train " + str(len(trainWsis)))
    print(" val " + str(len(valWsis)))
    print(" test " + str(len(testWsis)))

    valWsis = matchListLength(trainWsis,valWsis)
    testWsis = matchListLength(trainWsis,testWsis)

    for row in zip(trainWsis, valWsis, testWsis):
        print(row)

def makeTargetFolder(targetPath):
    makeFolderStructure(targetPath)
    

    trainPath = os.path.join(targetPath,"kryo","train")
    valPath = os.path.join(targetPath,"kryo","val")
    testPath = os.path.join(targetPath,"kryo","test")

    return trainPath, valPath, testPath

def copyTiles(tiles, inPath, destPath):

    for tile in tiles:
        
        diag = getDiagnosis(tile)
        safePath = os.path.join(destPath,diag)
        safePath = os.path.join(safePath,tile)
        srcPath = os.path.join(inPath,tile)
        #print(getDiagnosis(tile))
        shutil.move(srcPath,safePath)
        

        print(tile + " "+ destPath+"\n") 



def main(inPath, outPath):

    wsis = sortTilesByWSI(inPath)

    trainSet, valSet, testSet = split(wsis, 0.6, 0.3, 0.1)
    
    printSplit(trainSet , valSet , testSet)

    trainPath, valPath, testPath = makeTargetFolder(outPath)
   
    copyTiles(trainSet,inPath,trainPath)
    copyTiles(valSet,inPath,valPath)
    copyTiles(testSet,inPath,testPath)





if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--tilePath", help="The path to the folder containing the tiles")
    argParser.add_argument("-o", "--filePath", help="The path to the split folder")
   

    
    args = argParser.parse_args()

    inPath = args.tilePath
    outPath = args.filePath

    #inPath= r"C:\Users\felix\Desktop\neuroImages\kryo\train\GBM"

    #outPath = r"C:\Users\felix\Desktop\featureMap"

   

    

    main(inPath,outPath)


