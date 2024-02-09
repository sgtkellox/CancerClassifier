import os
import argparse
import shutil


def getSlideName(slide):
    slideName = slide.split("\\")[-1]
    slideName = slideName.split(".")[0]
    return slideName

def lookUpFolder(path, coreMap, thresh): 
    for folder in os.listdir(path):
        print("looking up" + folder)
        identifier = folder.split("-")[2]
        if identifier in coreMap.keys():
            folderPath = os.path.join(path,folder)
            for file in os.listdir(folderPath):
                if not file.endswith(".tif"):
                    continue
                else: 
                    if getSlideName(file) in coreMap[identifier]:
                        print("slide " + file + " is marked for removal")
                        srcPath = os.path.join(folderPath,file)
                        destPath = os.path.join(thresh,file)
                        shutil.move(srcPath,destPath)
            


def readFile(filePath):
    coresMap = {}
    with open(filePath) as f:
        lines = f.readlines()
        
    
    currentCore = ""
    for line in lines:
        if line.strip() == "":
            continue
        
        if line.startswith("TMA"):
            currentCore = line.strip().split("-")[1]
            cores = []
            coresMap[currentCore] = cores
           
                
            
        else:
            coresMap[currentCore].append(line.strip())
            
    print(coresMap)
    return coresMap

def removeBadCores(tablePath,slides,binPath):
    coreMap = readFile(tablePath)
    lookUpFolder(slides,coreMap,binPath )
    
    

            

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-c", "--cores", help="The Path to the cores")
    argParser.add_argument("-t", "--table", help="the path to the table")
    argParser.add_argument("-b", "--thresh", help="the path to the bin")
    
    
    args = argParser.parse_args()

    slides = args.cores
    tablePath = args.table
    binPath = args.thresh

    #path = r"C:\Users\felix\Downloads\not_representative_cores.txt"
    #pathImages = r"D:\coresTiff"
    coreMap = readFile(tablePath)
    lookUpFolder(slides,coreMap,binPath )
    
    
        
        
            
            
            


            
            
    
            