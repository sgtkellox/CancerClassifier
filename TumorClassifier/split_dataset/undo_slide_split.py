import os

import shutil

import argparse


def documentSplit(path,docPath):
    split = os.listdir(path)
    outpath = os.path.join(docPath,"doc.txt")

    for part in split:
        partPath = os.path.join(path,part)
        with open(outpath, 'a') as doc:
            print(part)
            doc.write(part)
            doc.write('\n')
            doc.write('------------------------')
            doc.write('\n')
            for folder_path, folders, slides in os.walk(partPath):
                for slide in slides:
                    doc.write(slide)
                    doc.write('\n')
            doc.write('------------------------')
            doc.write('\n')


def undoSplitFully(path, destPath):

    for prep in os.listdir(path):
        prepPath = os.path.join(path,prep)
        for part in os.listdir(prepPath):
            partPath = os.path.join(prepPath,part)
            for diag in os.listdir(partPath):
                diagPath = os.path.join(partPath,diag)           
                for slide in os.listdir(diagPath):
                    slidePath = os.path.join(diagPath,slide)
                    shutil.move(slidePath,os.path.join(destPath,slide))


def undoSplit(inPath,outPath):
    for prep in os.listdir(inPath):
        prepPath = os.path.join(inPath,prep)
        destPath = os.path.join(outPath,prep)
        for part in os.listdir(prepPath):
            partPath = os.path.join(prepPath,part)
            for diag in os.listdir(partPath):
                diagPath = os.path.join(partPath,diag)           
                for slide in os.listdir(diagPath):
                    slidePath = os.path.join(diagPath,slide)
                    shutil.move(slidePath,os.path.join(destPath,slide))



if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--inPath", help="The path to the folder containing the slides split")
    argParser.add_argument("-o", "--outPath", help="The path to the folder containing the unsplit slides")
    argParser.add_argument("-d", "--doc", help="The path whre u want the doc file to be safed")

    
    args = argParser.parse_args()

    inPath = args.inPath
    outPath = args.outPath
    docPath = args.doc


    #documentSplit(inPath,docPath)
    undoSplit(inPath,outPath)





    


