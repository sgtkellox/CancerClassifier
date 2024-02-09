import os 

import math
import shutil
import argparse 
import random

def readFile(filePath):
    
    with open(filePath) as f:
        lines = f.readlines()
          
        stripped = [s.strip() for s in lines]
        f.close()
    return stripped
           
                          
    

def makeSplit(lines, train, val, test):
   
    trainSet = []
    valSet = []
    
    testSet = []


    setSize = len(lines)

    valSize = math.floor(setSize*val)
    testSize = math.floor(setSize*test)
    
    x = 0
    
    while x<valSize:
        line = random.choice(lines)
        
        
        valSet.append(line)
        lines.remove(line)
        x +=1
        
    x = 0
    while x<testSize:
        line = random.choice(lines)
        
        
        testSet.append(line)
        lines.remove(line)
        x +=1

            
    for wsi in lines:
        
        trainSet.append(wsi)
        
    print(trainSet)
 
        
    return trainSet, valSet ,testSet

def makeGradeFolderStructure(parent):
    astroPath = os.path.join(parent,"one")
    gbmPath = os.path.join(parent,"two")
    

    diagPaths = []

    diagPaths.append(astroPath)
    diagPaths.append(gbmPath)
    

    for path in diagPaths:
        if not os.path.exists(path):
            os.makedirs(path)

def makeSplitFolderStructure(parent):
    trainPath = os.path.join(parent,"train")
    valPath = os.path.join(parent,"val")
    testPath = os.path.join(parent,"test")

    splitPaths = []

    splitPaths.append(trainPath)
    splitPaths.append(valPath)
    splitPaths.append(testPath)

    for path in splitPaths:
        if not os.path.exists(path):
            os.makedirs(path)
        makeGradeFolderStructure(path)
        
def getImageFronmLine(line):
    ident =  line.split(" ")[0]
    ident = ident + ".jpg"
    return ident

def getGradeFronmLine(line):
    ident =  line.split("   ")[1]
    print(ident)
    if ident == str(1):
        return "one"
    if ident == str(2) or ident == str(3): 
        return "two"
   
        
        
def copyTiles(inPath, destPath, lines):
    for line in lines:
        
        grade = getGradeFronmLine(line)
        
        slide = getImageFronmLine(line)
        safePath = os.path.join(destPath,grade)
        safePath = os.path.join(safePath,slide)
        
        srcPath = os.path.join(inPath,slide)
        
        if os.path.isfile(srcPath):
            shutil.copy(srcPath,safePath)
            
def split(inPath,filePath, outPath):
    lines = readFile(filePath)
    makeSplitFolderStructure(outPath)
    
    trainSet, valSet ,testSet = makeSplit(lines, 0.6, 0.3, 0.1)
    
    copyTiles(inPath, os.path.join(outPath,"train"), trainSet)
    copyTiles(inPath, os.path.join(outPath,"val"), valSet)
    copyTiles(inPath, os.path.join(outPath,"test"), testSet)
    
        

       

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--inPath", help="The path to the folder containing the slides split")
    argParser.add_argument("-f", "--filePath", help="The path wtih the split file")
    argParser.add_argument("-o", "--outPath", help="The path to where the new folder is supposed to go")

    args = argParser.parse_args()   
    
    inPath = args.inPath
    filePath = args.filePath
    outPath = args.outPath
    
    lines = readFile(filePath)
    makeSplitFolderStructure(outPath)
    
    trainSet, valSet ,testSet = makeSplit(lines, 0.6, 0.3, 0.1)
    
    copyTiles(inPath, os.path.join(outPath,"train"), trainSet)
    copyTiles(inPath, os.path.join(outPath,"val"), valSet)
    copyTiles(inPath, os.path.join(outPath,"test"), testSet)

   
    
    
    
    
    
    



    
    
    
    
    

    
    
    
    
