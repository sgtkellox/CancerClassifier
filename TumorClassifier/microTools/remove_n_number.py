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
    print("finished sorting by wsi")
    return wsis


def extractNNumbersFromFile(path):

    with open(path,"r") as f:
        lines = f.readlines()
    lines = [x.rstrip() for x in lines]
    lines = [x[1:] for x in lines]
    f.close()

    return lines


def extractNNumberFromWsi(wsi):
    split = wsi.split("-")
    nNumber = split[1]+"-"+split[2]
    return nNumber



def filterTiles(imagePath, filepath):

    wsis = sortTilesByWSI(imagePath)
    toRemove = extractNNumbersFromFile(filepath)
 
    counter = 0

  
    for wsi in wsis: 
        
        if wsi.startswith("Inf"): 
            print("Inf found in "+ wsi)
            continue
        nNumber = extractNNumberFromWsi(wsi)

        number = nNumber[1:]
        if not number[-1].isdigit():
            number= number[0:-1]
               
        if number in toRemove:
            print("wsi " + wsi + " with "+ str(len(wsis[wsi]))+" tiles")
            for image in wsis[wsi]:
                pathToSingleImage = os.path.join(imagePath,image)
                os.remove(pathToSingleImage)
           
            counter+=1
    print("total: " +str(counter))

    
        


        
if __name__ == '__main__':


    argParser = argparse.ArgumentParser()

    argParser.add_argument("-t", "--tilePath", help="The path to the folder containing the tiles")
    argParser.add_argument("-f", "--filePath", help="The path to the file with the N-Numbers to remove")
   

    
    args = argParser.parse_args()

    imagePath = args.tilePath
    filePath = args.filePath

 
    filterTiles(imagePath, filePath)







