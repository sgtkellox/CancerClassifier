import os 
import random
import math
import json
import argparse


def sortByDiagnosis(path):
    astros = []
    gbms = []
    oligos = []
    
    result = []
    
    for slide in os.listdir(path):
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

def makeEntry(slide):
    entry = {}
    if slide.startswith("A"):
        label = "A"
    elif slide.startswith("GBM"):
        label = "GBM"
    elif slide.startswith("O"):
        label = "O"

    entry["image"] = slide
    entry["label"] = label

    

    return entry

def suitedForVal(wsi):
    identifier = wsi.split("_")[0]
    num = identifier.split("-")[3]
    if num[-1].isdigit():
        return False
    else: 
        return True
    
    
        

def makeSplit(path, train, val):

    if val+train !=1:
        print("invalid split ratio")
        return

    trainSet = []
    valSet = []

    slides = os.listdir(path)

    setSize = len(slides)

    valSize = math.floor(setSize*val)
    
    x = 0
    
    while x<valSize:
        randomWsi = random.choice(slides)
        if suitedForVal(randomWsi):
            entry = makeEntry(randomWsi)
            valSet.append(entry)
            slides.remove(randomWsi)
            x +=1

            
    for wsi in slides:
        entry = makeEntry(wsi)
        trainSet.append(entry)


    splitDict = {}
    splitDict["training"] = trainSet
    splitDict["validation"] =  valSet
 
        
    return splitDict


if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--inPath", help="The path to the folder containing the slides split")
    argParser.add_argument("-f", "--filePath", help="The path to where the new folder is supposed to go")

    args = argParser.parse_args()   
    
    inPath = args.inPath
    filePath = args.filePath

    splitDict = makeSplit(inPath,0.7,0.3)
    
    jsonFile = json.dumps(splitDict,sort_keys=True,indent=4,separators=(',', ': '))

    with open(filePath, "w") as outfile:
        outfile.write(jsonFile)







