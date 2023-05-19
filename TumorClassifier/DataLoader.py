
import os
import shutil
import random
import math
from sysconfig import get_path

#from preprocessing_slides import process_tiles
#from preprocessing_slides import mask4
#from preprocessing_slides import SIZE

import matplotlib.pyplot as plt



project_name='BrainCancerClassifier'

GBMfolder = ""
oligoFolder = ""
astroFolder = ""


def getPreparationFromFileName(fileName):
    prepString = fileName.split("-")[3]
    prepString = fileName.replace(".svs","")
    if not prepString[-1].isdigit():
        return prepString[-1]
    else:
        return prepString[-2]


def getDiagnosisFromFileName(fileName):
    return fileName.split("-")[0]

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

    print(slides)

    for x in range(0,trainSize):
        chosenFile = random.choice(slides)
        trainSet.append(chosenFile)
        slides.remove(chosenFile)
    for x in range(0,valSize):
        chosenFile = random.choice(slides)
        valSet.append(chosenFile)
        slides.remove(chosenFile)
    for x in range(0,testSize):
        chosenFile = random.choice(slides)
        testSet.append(chosenFile)
        slides.remove(chosenFile)
    
    return trainSet, valSet, testSet



def sortByDiagnosis(splitPart,path,srcPath):
    

    astroPath = os.path.join(path,"Astro")
    gbmPath = os.path.join(path,"GBM")
    oligoPath = os.path.join(path,"Oligo")

    for slide in splitPart:
        if getDiagnosisFromFileName(slide) == "A":
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
    
    print("after kryo")

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

    





def makeDiagnosisFolderStructure(parent):
    astroPath = os.path.join(parent,"Astro")
    gbmPath = os.path.join(parent,"GBM")
    oligoPath = os.path.join(parent,"Oligo")

    diagPaths = []

    diagPaths.append(astroPath)
    diagPaths.append(gbmPath)
    diagPaths.append(oligoPath)

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
        makeDiagnosisFolderStructure(path)



def makePrepFolderStructure(path):
    kryoPath = os.path.join(path,"smear")
    smearPath = os.path.join(path,"kryo")
    touchPath = os.path.join(path,"touch")

    prepPaths = []

    prepPaths.append(kryoPath)
    prepPaths.append(smearPath)
    prepPaths.append(touchPath)

    for path in prepPaths:
        if not os.path.exists(path):
            os.makedirs(path)
        makeSplitFolderStructure(path)



def distributeLeftOverIntoTestSet(pathIn, pathOut):

    for slide in os.listdir(pathIn):
        print()
    

    
  


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
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Oligo\Touch")
            elif "Q" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Oligo\Smear")
            elif "K" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Oligo\Kryo")

        elif file.startswith("A"):
            if "T" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Astro\Touch")
            elif "Q" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Astro\Smear")
            elif "K" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Astro\Kryo")
    return


def sliceKryos(path):
     files = os.listdir(path)
     for file in files:
            if file.startswith("GBM"):
                
                if "K" in file:
                    #dst = os.path.join(GBMfolder,"Kryo",file)
                    filePath = os.path.join(path,file)
                    process_tiles(filePath,mask4, outPath=r"D:\splitTiles\kryo\test\GBM")
            elif file.startswith("O"):
                
                if  "K" in file:
                    filePath = os.path.join(path,file)
                    process_tiles(filePath,mask4, outPath=r"D:\splitTiles\kryo\train\Oligo")

            elif file.startswith("A"):
                
                if "K" in file:
                    filePath = os.path.join(path,file)
                    process_tiles(filePath,mask4, outPath=r"D:\KryoTiles\Astro")
     return   

def sortLeftOverSlides(path):
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
                process_tiles(filePath,mask4, outPath=r"D:\testSetsSmear\GBM")
            elif "K" in file:
                #dst = os.path.join(GBMfolder,"Kryo",file)
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"E:\ClassifierImages\GBM\Kryo")
        elif file.startswith("O"):
            if "T" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Oligo\Touch")
            elif "Q" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\testSetsSmear\Oligo")
            elif "K" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Oligo\Kryo")

        elif file.startswith("A"):
            if "T" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Astro\Touch")
            elif "Q" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\testSetsSmear\Astro")
            elif "K" in file:
                filePath = os.path.join(path,file)
                process_tiles(filePath,mask4, outPath=r"D:\ClassifierImages\Astro\Kryo")
    return


def sortSlidesByPreparation(inPath, kryoPath, smearPath, touchPath):
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


def splitSortedTiles(inPath,outPath):

    inAstro = os.path.join(inPath,"Astro","macenko","normalize")
    inGBM = os.path.join(inPath,"GBM","macenko","normalize")
    inOligo = os.path.join(inPath,"Oligo","macenko","normalize")
    
    wsisAstro = sortTilesByWSI(inAstro)
    wsisGBM = sortTilesByWSI(inGBM)
    wsisOligo = sortTilesByWSI(inOligo)
    print(wsisAstro)

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

    
        

if __name__ == '__main__':
    makePrepFolderStructure("/mnt/projects/neuropath_hd/data/split$")