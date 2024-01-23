import argparse
import os


def listFolder(path):
    split = os.listdir(path)
    

    trainSet = []

    valSet = []

    testSet = []

    for part in split:
        partPath = os.path.join(path,part)

        images = []
        
        for diag in os.listdir(partPath):
            diagPath = os.path.join(partPath,diag)
            for image in os.listdir(diagPath):
                images.append(image)
        if part == "train":
            trainSet= images.copy()
        elif  part == "val":
            valSet = images.copy()
        else:
            testSet = images.copy()

    return trainSet,valSet,testSet

               
       


def getWsiNames(trainSet, valSet , testSet):
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

    return trainWsis, valWsis, testWsis


def printWsis(trainWsis, valWsis, testWsis, outPath):

    with open(outPath, 'a') as doc:
            doc.write("train")
            doc.write('\n')
            doc.write('------------------------')
            doc.write('\n')
            for slide in trainWsis:
                doc.write(slide)
                doc.write('\n')
            doc.write("val")
            doc.write('\n')
            doc.write('------------------------')
            doc.write('\n')
            for slide in valWsis:
                doc.write(slide)
                doc.write('\n')
            doc.write("test")
            doc.write('\n')
            doc.write('------------------------')
            doc.write('\n')
            for slide in testWsis:
                doc.write(slide)
                doc.write('\n')
            doc.close()







if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-p", "--path", help="The Path where ur datasplit is stored")
    argParser.add_argument("-f", "--file", help="the path where the outfile is stored")

    args = argParser.parse_args()

    path = args.path
    docPath = args.file

    t,testSet = listFolder(path)
    trainWsis, valWsis, testWsis = getWsiNames(trainSet,valSet,testSet)
    printWsis(trainWsis, valWsis, testWsis, docPath)

