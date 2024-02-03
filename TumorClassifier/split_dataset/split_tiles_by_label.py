import os
import argparse

import shutil 
import random

import math

from label_folder_structure import makeFolderStructure

from image_utils.tile_utils import getPreparation

glialList = ["A", "O", "GBM" , "PXA" , "PA", "H3"]
non_glialList = ["MB", "MET", "LYM", "MEL" ,"MEN" , "SCHW", "PIT"]



def getGroupByLabel(tile, label):
    
    if label == "diag":
        if tile.startswith("A"):
            return "Astro"
        elif tile.startswith("GBM"):
            return "GBM"
        elif tile.startswith("O"):
            return "Oligo"
        else:
            return tile.split("-")[0]
        
    elif label == "grade":
        diagPart = tile.split("-")[0]
        if diagPart.startswith("GBM"):
            return "four"
        elif diagPart[-1] == str(2):
            return "two"
        elif diagPart[-1] == str(3):
            return "three"
        elif diagPart[-1] == str(4):
            return "four"
        
    elif label == "gradeR":
        diagPart = tile.split("-")[0]
        if diagPart.startswith("GBM"):
            return "high"
        elif diagPart[-1] == str(2):
            return "low"
        elif diagPart[-1] == str(3):
            return "high"
        elif diagPart[-1] == str(4):
            return "high"
            
    elif label == "diff": 
        diagPart = tile.split("-")[0]
        if tile.startswith("O"):
            return "Oligo"       
        else:
            return "Astro"
        
    elif label == "idh":
        diagPart = tile.split("-")[0]
        if tile.startswith("GBM"):
            return "wild"
        else:
            return "mutated"
        
    elif label == "glial":
        diagPart = tile.split("-")[0]
        if diagPart in glialList:
            return "glial"
        elif diagPart in non_glialList:
            return "non-glial"
        
    
        


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
    
def split(labels, trainRatio, valRatio, testRatio):
    

    trainSet = []
    valSet = []
    testSet = []
    
    #print(labels)
    

    for label in labels:
        
        
        setSize = len(label)
  

        valSize = math.floor(setSize*valRatio)

        testSize = math.floor(setSize*testRatio)
        
        x = 0
    
        while x<valSize:
            randomWsi = random.choice(label)
            if suitedForVal(randomWsi):
                valSet.append(randomWsi)
                label.remove(randomWsi)
                x +=1

        x = 0
        while x < testSize:
            randomWsi  = random.choice(label)
            if suitedForVal(randomWsi):
                testSet.append(randomWsi)
                label.remove(randomWsi)
                x +=1
    
            
        for wsi in label:
            trainSet.append(wsi)
        
    return trainSet , valSet, testSet
    

def matchListLength(longList, shortList):
    diff = len(longList)-len(shortList)
    diffList = ["---"]*diff

    shortList = shortList + diffList
    return shortList

    
def printSplit( trainSet, valSet , testSet, outPath):

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
    
    with open(outPath, 'a') as doc:
        

        for row in zip(trainWsis, valWsis, testWsis):
            #print(row) 
            doc.write(str(row))
            doc.write("\n")
        
        
    

def sortByDiagnosis(wsis):
    astros = []
    gbms = []
    oligos = []
    
    result = []
    
    for slide in wsis:
        if slide.startswith("A"):
            astros.append(slide)
        elif slide.startswith("GBM"):
            gbms.append(slide)
        elif slide.startswith("O"):
            oligos.append(slide)
            
    result.append(astros)
    result.append(gbms)
    result.append(oligos)
    
    return result

def sortByDiag(wsis):
    mb = []
    lym = []
    met = []
    mel = []
    men = []
    schw = []
    pit = []
    
    result = []
    
    for slide in wsis:
        if slide.startswith("MB"):
            mb.append(slide)
        elif slide.startswith("LYM"):
            lym.append(slide)
        elif slide.startswith("MET"):
            met.append(slide)
        elif slide.startswith("MEL"):
            mel.append(slide)
        elif slide.startswith("MEN"):
            men.append(slide)
        elif slide.startswith("SCHW"):
            schw.append(slide)
        elif slide.startswith("PIT"):
            pit.append(slide)
            
    result.append(mb)
    result.append(lym)
    result.append(met)
    result.append(mel)
    result.append(men)
    result.append(schw)
    result.append(pit)
    
    return result
    
    
    
    


def sortByGrade(wsis):
    
    two = []
    three = []
    four = []
    
    result = []
    
    for slide in wsis:
        diagPart = slide.split("-")[0]
        if diagPart.startswith("GBM"):
            four.append(slide)
        elif diagPart[-1] == str(2):
            two.append(slide)
        elif diagPart[-1] == str(3):
            three.append(slide)
        elif diagPart[-1] == str(4):
            four.append(slide)
            
    result.append(two)
    result.append(three)
    result.append(four)
    
    return result

def sortByRoughGrade(wsis): 
    
    high = []
    low = []
    
    
    result = []
    
    for slide in wsis:
        diagPart = slide.split("-")[0]
        if diagPart.startswith("GBM"):
            high.append(slide)
        elif diagPart[-1] == str(2):
            low.append(slide)
        elif diagPart[-1] == str(3):
            high.append(slide)
        elif diagPart[-1] == str(4):
            high.append(slide)
            
    result.append(high)
    result.append(low)
    
    return result
    

def sortByDifferentiation(wsis):
    
    astros = []
    oligos = []
    
    result = []
    
    for slide in wsis:
        diagPart = slide.split("-")[0]
        if slide.startswith("A"):
            astros.append(slide)
        elif slide.startswith("GBM"):
            astros.append(slide)
        elif slide.startswith("O"):
            oligos.append(slide)
        
            
    result.append(astros)
    result.append(oligos)
    
    return result
    
    
def sortByIDHstatus(wsis):
    mutated = []
    wild = []
    
    result = []
    
    for slide in wsis:
        diagPart = slide.split("-")[0]
        if slide.startswith("A"):
            mutated.append(slide)
        elif slide.startswith("GBM"):
            wild.append(slide)
        elif slide.startswith("O"):
            mutated.append(slide)
                
    result.append(mutated)
    result.append(wild)
    
    return result

def sortByLocalisation(wsis):
    glial = []
    nonglial = []
    
    result = []
    
    for slide in wsis:
        diagPart = slide.split("-")[0]
        if diagPart in glialList:
            glial.append(slide)
        elif diagPart in non_glialList:
            nonglial.append(slide)
                       
    result.append(glial)
    result.append(nonglial)
    
    return result
        
    


def copyTiles(inPath, outPath, label, trainSet, valSet, testSet, wsis):
    
    trainPath = os.path.join(outPath,"train")
    valPath = os.path.join(outPath,"val")
    testPath = os.path.join(outPath,"test")
    
    for item in trainSet:       
        itemLabel = getGroupByLabel(item,label)
        itemPath = os.path.join(trainPath,itemLabel)
        for image in wsis[item]:
            srcPath = os.path.join(inPath,image)
            destPath = os.path.join(itemPath,image)
            shutil.copy(srcPath,destPath)
    for item in valSet:       
        itemLabel = getGroupByLabel(item,label)
        itemPath = os.path.join(valPath,itemLabel)
        for image in wsis[item]:
            srcPath = os.path.join(inPath,image)
            destPath = os.path.join(itemPath,image)
            shutil.copy(srcPath,destPath)
    for item in testSet:       
        itemLabel = getGroupByLabel(item,label)
        itemPath = os.path.join(testPath,itemLabel)
        for image in wsis[item]:
            srcPath = os.path.join(inPath,image)
            destPath = os.path.join(itemPath,image)
            shutil.copy(srcPath,destPath)
            
            
    
    

def printResult(result):
    for entry in result:
        print("--------------")
        for file in entry:
            print(file)
            

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--inPath", help="The path to the folder containing the slides")
    argParser.add_argument("-o", "--outPath", help="The path to where the split is supposed to go")
    argParser.add_argument("-a", "--attribute", help="The Atrribute to split by")
    
    
    
    args = argParser.parse_args()

    attr = args.attribute
    
    inPath = args.inPath
    
    outPath = args.outPath
    
    wsis = sortTilesByWSI(inPath)
    
    exampleSlide = list(wsis.keys())[0]
    
    prep = getPreparation(exampleSlide)
    
    
    diagList = ["MB", "MET", "LYM", "MEL" ,"MEN" , "SCHW", "PIT"]
   
    
    if attr == "diag":
        result = sortByDiag(wsis)
        folders = diagList
        
    elif attr == "idh":
        result = sortByIDHstatus(wsis)
        folders = {"wild","mutated"}
        
    elif attr == "diff":
        result = sortByDifferentiation(wsis)
        folders = {"Astro","Oligo"}
        
    elif attr == "grade":
        result = sortByGrade(wsis)
        folders = {"two","three","four"}
    
    elif attr == "gradeR":
        result = sortByRoughGrade(wsis)
        folders = {"low","high"}
    elif attr == "glial":
        result = sortByLocalisation(wsis)
        folders = {"glial","non-glial"}
        
             
 
    trainSet , valSet, testSet = split(result,0.6 ,0.3,0.1)
    
    makeFolderStructure(outPath,folders)
    
    filePath = os.path.join(outPath,"splitDoc.txt")
    
    outPath = os.path.join(outPath,prep)
    
    copyTiles(inPath, outPath, attr, trainSet, valSet, testSet, wsis)
    
    
    printSplit(trainSet , valSet, testSet, filePath)
    
    
    
    