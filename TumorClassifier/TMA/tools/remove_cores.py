import os


def getSlideName(slide):
    slideName = slide.split("\\")[-1]
    slideName = slideName.split(".")[0]
    return slideName

def lookUpFolder(path, coreMap):
    for folder in os.listdir(path):
        if folder in coreMap.keys():
            folderPath = os.path.join(path,folder)
            for file in os.listdir(folderPath):
                if not file.endwith(".tif"):
                    continue
                else: 
                    if getSlideName(file) in coreMap[getSlideName(file)]:
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
            currentCore = line.strip()
            cores = []
            coresMap[currentCore] = cores
           
                
            
        else:
            coresMap[currentCore].append(line.strip())
            
    print(coresMap)
    

            

if __name__ == '__main__':
    path = r"C:\Users\felix\Downloads\not_representative_cores.txt"
    pathImages = r""
    coreMap = readFile(path)
    lookUpFolder(pathImages,coreMap )
    
    
        
        
            
            
            


            
            
    
            