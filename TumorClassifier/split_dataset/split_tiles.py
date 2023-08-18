import os
import argparse
import math
import random


def sortTilesByWSI(path):

    wsis = {}

    for img in os.listdir(path):

        wsiName = img.split("_")[0]

        if wsiName in wsis:
            wsis[wsiName].append(img)
        else:
            wsis[wsiName] = []
            wsis[wsiName].append(img)
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



def split(wsis, trainRatio, valRatio, testRation):

    setSize = len(wsis)

    valSize = math.floor(setSize*trainRatio)

    testSize = math.floor(setSize*trainRatio)

    trainSet = []
    valSet = []
    testSet = []

    x = 0
    while x<valSize:
        randomWsi = random.choice([wsis.keys()])
        if suitedForVal(randomWsi):
            valSet = valSet + wsis[randomWsi]
            del wsis[]
            x +=1

    x = 0
    while x < testSize:
        randomWsi = random.choice([wsis.keys()])
        if suitedForVal(randomWsi):
            testSet = testSet + wsis[randomWsi]
            x +=1






    

    

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-t", "--tilePath", help="The path to the folder containing the tiles")
    argParser.add_argument("-f", "--filePath", help="The path to the result file")
   

    
    args = argParser.parse_args()

    imagePath = args.tilePath
    filePath = args.filePath


