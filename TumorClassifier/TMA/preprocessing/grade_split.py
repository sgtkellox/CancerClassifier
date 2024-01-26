import os
import argparse

from color_deconvolution import getSlideName, processFile



def mixImages(pathIn, pathOut):
    
    bwPath = os.path.join(pathOut,"grey")
    rgbPathOut = os.path.join(pathOut,"rgb")
    
    os.mkdir(bwPath)
    os.mkdir(rgbPathOut)
   
    for folder in os.listdir(pathIn):
        print("looking up" + folder)
        identifier = folder.split("-")[2]        
        folderPath = os.path.join(pathIn,folder)
        for file in os.listdir(folderPath):
            if not file.endswith(".tif"):
                continue
            else:
                newFileName = os.path.join(rgbPathOut, str(identifier)+"-"+getSlideName(file)+".jpg")
                filePath = os.path.join(folderPath,file)
                greyPath = os.path.join(bwPath,str(identifier)+"-"+getSlideName(file)+".jpg")
                print(greyPath)
                processFile(filePath,newFileName,greyPath)
                
                
                


if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--inPath", help="The Path to the cores")
    argParser.add_argument("-o", "--outPath", help="the path where the result goes")
   
    args = argParser.parse_args()

    inPath = args.inPath
    outPath = args.outPath
    
    mixImages(inPath,outPath)
    
    
                
                    