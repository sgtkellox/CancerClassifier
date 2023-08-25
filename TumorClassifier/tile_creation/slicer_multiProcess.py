
import os



import multiprocessing

import argparse

from make_tiles import make_tiles


def sliceSubset(pathIn, pathOut, slides,size):
    for slide in slides:
        if slide.endswith(".svs"):
             filePath = os.path.join(pathIn,slide)      
             make_tiles(filePath, outPath=pathOut, size = size)

def printInfo(subSet):
    print("started processing subset" +str(i)+ " of " + str(cpus)+"\n")
    print(" tiling subset with:")
    for slide in subSet:
        print("slideName" + slide)




if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--input", help="The path to the folder containing the slides")
    argParser.add_argument("-o", "--out", help="The path to the folder where u want the tiles")
    argParser.add_argument("-s", "--size",type=int, default=500, help="size of the tiles in pixel")

    args = argParser.parse_args()

    pathIn = args.input
  
    pathOut = args.out

    size = args.size

    cpus = multiprocessing.cpu_count()-4

    print("Number of cpu : ", cpus)

    images = os.listdir(pathIn)

    numImages = len(images)

    procs = []
    cpus = int(cpus)

    #createOutPath(pathOut)

    for i in range(1,cpus+1):
        subSet = images[int((i-1)*numImages/cpus):int(i*numImages/cpus)]
        printInfo(subSet)
        tilingProc = multiprocessing.Process(target=sliceSubset,args=(pathIn,pathOut,subSet,size))
        procs.append(tilingProc)
        tilingProc.start()
        

    for process in procs:
        process.join()
        print("folder finished")

    