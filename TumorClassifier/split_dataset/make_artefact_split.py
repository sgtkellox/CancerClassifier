import os
import math
import random
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

    #print(wsis)
    return wsis


def split(images, valRatio):
    
    trainSet = []
    valSet = []
    
    setSize = len(images)

    print(len(images))
  

    valSize = math.floor(setSize*valRatio)

        
           
    x = 0
    
    while x<valSize:
        randomWsi = random.choice(images)
           
        valSet.append(randomWsi)
        images.remove(randomWsi)
        x +=1

      
    
            
    for wsi in images:
        trainSet.append(wsi)
    
    print("train")
    print(trainSet)
    print("val")

    print(valSet)

    
        
    return trainSet , valSet

def makeResultPath(outPath):
    trainPath =  os.path.join(outPath, "train")
    valPath =  os.path.join(outPath, "val")

    for entry in [trainPath,valPath]:
        path = os.path.join(entry,"Artefact")
        if not os.path.isdir(path):
            os.makedirs(path)
        path = os.path.join(entry,"Good")
        if not os.path.isdir(path):
            os.makedirs(path)


def copyTiles(inPath, outPath, trainSet, valSet, wsis, label):
    
    if label == "a":
        srcPath = os.path.join(inPath, "Artefact")
        destTrain = os.path.join(outPath,"train", "Artefact")
        destVal = os.path.join(outPath,"val", "Artefact")
    elif label == "g":
        srcPath = os.path.join(inPath, "Good")
        destTrain = os.path.join(outPath,"train", "Good")
        destVal = os.path.join(outPath,"val", "Good")
    

    for wsi in trainSet:
        #print(wsi)
        for image in wsis[wsi]:
            imgSrcPath = os.path.join(srcPath, image)
            imgDestPath = os.path.join(destTrain, image)
            shutil.move(imgSrcPath,imgDestPath)

            #print(image)
    for wsi in valSet:
        #print(wsi)
        for image in wsis[wsi]:
            imgSrcPath = os.path.join(srcPath, image)
            imgDestPath = os.path.join(destVal, image)
            shutil.move(imgSrcPath,imgDestPath)
            #print(image)
        
       



def main(inPath, outPath):
    artefactPath = os.path.join(inPath,"Artefact")
    goodPath = os.path.join(inPath,"Good")

    artefactWsis  = sortTilesByWSI(artefactPath)
    goddWsis  = sortTilesByWSI(goodPath)
    
    makeResultPath(outPath)

    

    artefactTrain, artefactVal = split(list(artefactWsis.keys()), 0.3)
    goodTrain, goodVal = split(list(goddWsis.keys()), 0.3)

    copyTiles(inPath, outPath, artefactTrain, artefactVal, artefactWsis, "a")
    copyTiles(inPath, outPath, goodTrain, goodVal, goddWsis, "g")



if __name__ == '__main__':

    inPath = r"C:\Users\felix\Desktop\AutoEncoder\Tiles"
    outPath = r"C:\Users\felix\Desktop\AutoEncoder\artefactSplit"
    main(inPath,outPath)