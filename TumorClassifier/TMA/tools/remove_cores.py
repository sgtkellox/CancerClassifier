import os


def lookUpFolder(path, slide, cores):
    for folder in os.listdir(path):
        if folder == slide:
            folderPath = os.path.join(path,folder):
            


def readFile(filePath):
    coresMap = {}
    with open(filePath) as f:
        lines = f.readlines()
        
    cores = []
    
    for line in lines:
        if line.strip() == "":
            continue
        
        if line.startswith("TMA"):
            newCore = line
            if len(cores) != 0:
                
            cores.clear()
        else:
            cores.append(line)
        
        
            
            
            


            
            
    
            