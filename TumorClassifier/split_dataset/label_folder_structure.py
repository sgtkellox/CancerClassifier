import os

import argparse



            

def makeLabelFolderStructure(parent, labels):
    labelPaths = []
    for label in labels:
        path = os.path.join(parent,label)
        labelPaths.append(path)
    
    for path in labelPaths:
        if not os.path.exists(path):
            os.makedirs(path)


def makeSplitFolderStructure(parent,labels):
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
        makeLabelFolderStructure(path,labels)



def makeFolderStructure(path,labels):
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
        makeSplitFolderStructure(path,labels)



if __name__ == '__main__': 

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-p", "--path", help="The path to the folder IN which u want the Folder Structure to be created")
    argParser.add_argument("-l", "--labels", help="Labels to sort by")
    

    args = argParser.parse_args()
    

    
    

    path = args.path
    labels = args.labels
    
    if labels == "diag":
        folders = {"Astro","GBM","Oligo"}
        
    elif labels == "idh":
        folders = {"wild","mutated"}
        
    elif labels == "diff":
        folders = {"Astro","Oligo"}
    
    elif labels == "grade":
        folders = {"two","three","four"}
    
    elif labels == "gradeR":
        folders = {"low","high"}

    makeFolderStructure(path,folders)
