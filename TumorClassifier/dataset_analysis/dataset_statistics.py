import os 
import pandas as pd 

def extractNNumberFromWsi(wsi):
    wsi = wsi.split(".")[0]
    split = wsi.split("-")
    nNumber = split[1]+"-"+split[2]
    return nNumber


def extraxtPrepInfo(wsi):
    print(wsi)
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
    
def collectNNumbersFromTable(table,nToDiagMap):
       
    
    
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
        elif diag == "Glioblastoma":
            app = "GBM"
        elif diag == "Astrocytoma":
            app = "A"
        elif diag == "Oligodendroglioma":
            app = "O"
            
        nToDiagMap[nNumber] = app
        
    return nToDiagMap 

def collectDiagFullNameFromTable(table):
    nToDiagMap = {}
    
   
    
    for index, row in table.iterrows():
        nNumber = row["Patho-Nr"]
        nNumber = nNumber.strip()
        
        if " " in nNumber:
            nNumber = nNumber.split(" ")[0]
            
        
        diag = row["Diagnosis"]
                       
        nToDiagMap[nNumber] = diag
        
    return nToDiagMap 
    

def listNNumbersInTable(table, numbersMap):
    
    for index, row in table.iterrows():
        nNumber = row["Patho-Nr"]
        nNumber = nNumber.strip()
        
        if " " in nNumber:
            nNumber = nNumber.split(" ")[0]
            
        prepMap = {"K":0,"Q":0,"T":0}
        
        numbersMap[nNumber] = prepMap
        
    return numbersMap

def lookUpNnumber(nNumber,paths):
    matchingSlides = []
    for path in paths:
        
        for slide in os.listdir(path):
            
            if not slide.endswith(".svs"):
                continue
            slideNumber = extractNNumberFromWsi(slide)
            
            if slideNumber == nNumber:
                
                matchingSlides.append(slide)
    return matchingSlides


def fillMapFromFiles(numbersMap,path):
    for number in numbersMap.keys():
        matchingSlides = lookUpNnumber(number,path)
        for slide in matchingSlides:
                    
            prep = extraxtPrepInfo(slide)
            numbersMap[number][prep] = 1
    return numbersMap


def mapToPandas(numbersMap, excelPath,diagsMap):
    numbers = []
    kryos = []
    smears = []
    touch = []
    diags = []
    
    
    for number in numbersMap.keys():
        diags.append(diagsMap[number])
        numbers.append(number)
        if numbersMap[number]["K"] ==1:
            kryos.append("yes")
        else:
            kryos.append("no")
            
        if numbersMap[number]["Q"] ==1:
            smears.append("yes")
        else:
            smears.append("no")
            
        if numbersMap[number]["T"] ==1:
            touch.append("yes")
        else:
            touch.append("no")
            
    data = {
        'NNummer': numbers,
        'Diag': diags,
        'Kryo': kryos,
        'Smear': smears,
        'Touch' : touch
    }
    print("num " + str(len(numbers)))
    print("diags " + str(len(diags)))
    print("Kryo " + str(len(kryos)))
    print("Smear " + str(len(smears)))
    print("Touch " + str(len(touch)))

    df = pd.DataFrame(data)
    df.to_excel(excelPath, index=True)
    

    
def makeExcel(tablePaths, filePaths, outPath):
    nToDiagMap = {}
    prepMap = {}
    for tablePath in tablePaths:
        table = pd.read_excel(tablePath)
        
        nToDiagMap = collectNNumbersFromTable(table,nToDiagMap)
        
        prepMap = listNNumbersInTable(table, prepMap)
        
        print(nToDiagMap)
        
        numbersMapFilled = fillMapFromFiles(prepMap,filePaths)
    mapToPandas(numbersMapFilled, outPath,nToDiagMap)
    
    
    
        
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
    
    path1 = r"D:\slides\kryoQ2"
    path2 = r"E:\newEntities"
    path3 = r"D:\slides\kryoQ1"
    path4 = r"D:\slides\smear"
    path5 = r"D:\slides\touch"
    
    
   
    
    
    
    tablePath1 = r"C:\Users\felix\Downloads\All_New_Cases_SS_11_entities(1).xlsx"
    tablePath2 = r"C:\Users\felix\Downloads\All_glioma_SS.xlsx"
    
    outPath = r"E:\out\overview4.xlsx"
    
    paths = []
    
    tablePaths = []
    
    tablePaths.append(tablePath1)
    tablePaths.append(tablePath2)
    
    paths.append(path1)
    paths.append(path2)
    paths.append(path3)
    

    #groupNNumbersByDiag(path,"K")
    makeExcel(tablePaths, paths,outPath)
    
    
    

