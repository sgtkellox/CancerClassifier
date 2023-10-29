import os

import argparse


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



def makeFolderStructure(path):
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

if __name__ == '__main__': 

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-p", "--path", help="The path to the folder IN which u want the Folder Structure to be created")

    args = argParser.parse_args()

    path = args.path

    makeFolderStructure(path)


