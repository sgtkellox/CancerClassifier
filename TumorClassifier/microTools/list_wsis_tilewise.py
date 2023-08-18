import os
import argparse



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


def printWsiNamesToFile(wsis, outPath):

    slideList = list(wsis.keys())
    slideList.sort()

    with open(outPath, "a") as res:
        for wsi in slideList:
            res.write(wsi)
            res.write("\n")
        res.close()



if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-t", "--tilePath", help="The path to the folder containing the tiles")
    argParser.add_argument("-f", "--filePath", help="The path to the result file")
   

    
    args = argParser.parse_args()

    imagePath = args.tilePath
    filePath = args.filePath

    wsis = sortTilesByWSI(imagePath)
    printWsiNamesToFile(wsis,filePath)

    