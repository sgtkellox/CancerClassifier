import os
import argparse


def getSlideName(slide):
    slideName = slide.split("\\")[-1]
    slideName = slideName.split(".")[0]
    return slideName

def lookUpFolder(path, coreMap):
   
    for folder in os.listdir(path):
        identifier = folder.split("-")[2]
        if identifier in coreMap.keys():
            folderPath = os.path.join(path,folder)
            for file in os.listdir(folderPath):
                if not file.endswith(".tif"):
                    continue
                else: 
                    if getSlideName(file) in coreMap[identifier]:
                        print("slide " + file + " is marked for removal")
            


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
    

            

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-c", "--cores", help="The Path to the cores")
    argParser.add_argument("-t", "--table", help="the path to the table")
    
    
    args = argParser.parse_args()

    slides = args.cores
    tablePath = args.table

    path = r"C:\Users\felix\Downloads\not_representative_cores.txt"
    pathImages = r"D:\coresTiff"
    coreMap = readFile(path)
    lookUpFolder(pathImages,coreMap )
    
    
        
        
            
            
            


            
            
    
            