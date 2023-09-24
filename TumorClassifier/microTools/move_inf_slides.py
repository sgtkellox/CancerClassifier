import os
import argparse
import shutil




def extractNNumberFromWsi(wsi):
    wsi = wsi.split(".")[0]
    split = wsi.split("-")
    nNumber = split[1]+"-"+split[2]
    return nNumber


def extractNNumbersFromFile(path):

    with open(path,"r") as f:
        lines = f.readlines()
    lines = [x.rstrip() for x in lines]
    lines = [x.replace("-"+x.split("-")[2],"") for x in lines]
    
    
    f.close()

    

    return lines


def filterTiles(slidePath, filepath, outpath):

    
    toRemove = extractNNumbersFromFile(filepath)
 
    counter = 0

  
    for wsi in os.listdir(slidePath): 
        
        nNumber = extractNNumberFromWsi(wsi)
        #print(nNumber)
 
               
        if nNumber in toRemove:
            #print("wsi " + wsi)
            pathToSingleImage = os.path.join(imagePath,wsi)
            #print(pathToSingleImage)
            pathToImgDest = os.path.join(outpath,wsi)

            print(pathToImgDest)

            
            shutil.move(pathToSingleImage,pathToImgDest)
            
                
           
            counter+=1
    print("total: " +str(counter))


if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-t", "--tilePath", help="The path to the folder containing the tiles")
    argParser.add_argument("-f", "--filePath", help="The path to the file with the N-Numbers to remove")
    argParser.add_argument("-o", "--outPath", help="The path to the dir, u want the tiles to be moved to")


    
   

    
    args = argParser.parse_args()

    imagePath = args.tilePath
    filePath = args.filePath
    outPath = args.outPath

 
    filterTiles(imagePath, filePath, outPath)