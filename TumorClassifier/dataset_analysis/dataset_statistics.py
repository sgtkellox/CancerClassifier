import os 
import pandas as pd 

def extractNNumberFromWsi(wsi):
    wsi = wsi.split(".")[0]
    split = wsi.split("-")
    nNumber = split[1]+"-"+split[2]
    return nNumber


def extraxtPrepInfo(wsi):
    wsi = wsi.split(".")[0]
    split = wsi.split("-")
    prep = split[3]
    if prep[-1].isdigit():
        return prep[0]
    else: 
        return prep

def extractDiag(wsi):
    split = wsi.split("-")
    
    if split[0].startswith("A"):
        return "A"
    
    elif split[0].startswith("O"):
        return "O"
    
    else:
        return split[0]
    
        
        
    



def groupNNumbersByDiag(path,prep):
    
    diags = ["PA", "PXA", "MET", "MB", "MEL", "MEN", "H3", "A", "O", "GBM", "PIT" , "LYM", "SCHW", "EPN" ]
    
    #counts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    diagsMap = {}
    
    for diag in diags:
        diagsMap[diag] = 0
    
    
    for folder in os.listdir(path):
        
        folderPath = os.path.join(path,folder)
        for slide in os.listdir(folderPath):
            
            if slide.endswith(".svs"):
                
                if prep == extraxtPrepInfo(slide):
                
                    diag = extractDiag(slide)
                    diagsMap[diag] += 1
    
    
    
    
    
    
                
        
    print(list(diagsMap.keys()))
    print(list(diagsMap.values()))
    

def compareWithList(slidesPath,listPath):
    slidesInTable= {}
    
    slidesInFolder={}
    
def collectNNumbersFromTable(tablePath):
    
    nToDiagMap = {}
    
    table = pd.read_excel(tablePath)
    
    for index, row in table.iterrows():
        nNumber = row["Patho-Nr"]
        nNumber = nNumber.strip()
        
        if " " in nNumber:
            nNumber = nNumber.split(" ")[0]
            
        
        diag = row["Diagnosis"]
            
        if diag == "PXA":
            app = "PXA"
        elif diag == "Pilocytic astrocytoma":
            app = "PA"     
        elif diag == "Metastasis":
            app = "MET"    
        elif diag == "Medulloblastoma" or diag == "Medulloblastom":
            app = "MB"
        elif diag == "Lymphoma" or diag == "Lymphom":
            app = "LYM"
        elif diag == "Ependymoma":
            app = "EPN"
        elif diag == "Neurinom":
            app = "SCHW"
        elif diag == "Hypophysenadenom":
            app = "PIT"
        elif diag == "H3-mutiertes Gliom":
            app = "H3"
        elif diag == "Meningeoma":
            app = "MEN"
        elif diag == "Melanom":
            app = "MEL"
            
        nToDiagMap[nNumber] = app
        
    return nToDiagMap 


def collectNNumbersFromFiles(path):
    
    diagsMap = {}
    for folder in os.listdir(path):
        
        folderPath = os.path.join(path,folder)
        for slide in os.listdir(folderPath):
            
            if slide.endswith(".svs"):
                
                nNumber = extractNNumberFromWsi(slide)
                diag = extractDiag(slide)
                diagsMap[nNumber] = diag
    return diagsMap
                
                
                
    
def printList(numbers, path):
    with open(path, 'a') as doc:
        for number in numbers:
            doc.write(number)
            doc.write('\n')
    doc.close()
       
        
    
        
    
    

def checkCorrectDiags(slidesPath, tablePath, outPath):
    
    nToDiagTable = collectNNumbersFromFiles(slidesPath)
    nToDiagFolder = collectNNumbersFromTable(tablePath)
    
    approvedPath = os.path.join(outPath,"approved.txt")
    notFoundPath = os.path.join(outPath,"notFound.txt")
    wrongPath = os.path.join(outPath,"wrong.txt")
    
    approved = []
    missing = []
    wrong = []
    

    for entry in nToDiagFolder.items():

        if entry[0] in nToDiagTable:
            if nToDiagTable[entry[0]]== entry[1]:
                approved.append(entry[0])
            else:
                wrong.append(entry[0])
        else:
            missing.append(entry[0])
            

    print(missing)
           

    printList(approved,approvedPath)
    printList(missing,notFoundPath)
    printList(wrong,wrongPath)
            

    
               
                    
                
    
    
    
    
def printMap(map):
    for entry in map.items():
        print(entry)
        
                
                
                
                


if __name__ == '__main__':
    
    path = r"D:\slides"
    
    tablePath = r"C:\Users\felix\Downloads\All_New_Cases_SS_11_entities(1).xlsx"
    
    outPath = r"D:\out"
    

    #groupNNumbersByDiag(path,"K")
    checkCorrectDiags(path, tablePath,outPath)
    
    
    

