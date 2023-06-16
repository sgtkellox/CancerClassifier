import os
import shutil

import pandas as pd

import math
#from openpyxl import load_workbook


def copySlides():
    for index, row in table1.iterrows():


        nNumber = row['txt_PATHOLOGIE_NUMMER']

        uuid = row['uuid']

        if " " in nNumber:
            #print(nNumber)
            continue
        if str(uuid) == 'nan':
            continue

        diagnosis = row['txt_TUMOR_450K_lang']

        slideName = str(uuid)+".svs"
        slidePath = os.path.join(pathToDB,slideName)

        if not os.path.isfile(slidePath):
            
            continue


        if "glioblastoma" in diagnosis:
            diagnosisChar = "GBM"
        elif "astrocytoma" in diagnosis:
            diagnosisChar = "A"
        elif "oligodendroglioma" in diagnosis:
            diagnosisChar = "O"
        
        
        slideDestName = diagnosisChar+"-"+nNumber+"-" + str(index)+".svs"
        slideDestPath = os.path.join(pathOut,slideDestName)

        if slideName in os.listdir(pathToDB):
            print(str(uuid))
        
            shutil.copyfile(slidePath,slideDestPath)

        

def checkForCorrectDiagnosis():
    return


def extractNNumberFromSlide(slide):

    parts = slide.split("-")
    nNumber = parts[1] + "-" + parts[2]

    return nNumber


def checkIfFileIsAllreadyCopied(nNumber):



    return


def fixRaminsStupidNameConvention(path):

    for slide in os.listdir(path):
        
        fileNameIn = os.path.join(path,slide)
        frontPart = slide[0:len(slide)-5]
        backPart = slide[len(slide)-5:len(slide)]
        newName = frontPart + "-" +backPart
        
        fileNameOut = os.path.join(path,newName)

        os.rename(fileNameIn, fileNameOut)

    return

 

def findFilesWithWrongConvention(path):

    for slide in os.listdir(path):
        preparation = slide.split("-")[3].split(".")[0]

        if preparation.isdigit(): 
            print(slide)



def collectCaseNumbersInTable(tablePath):
   
    table = pd.read_excel(tablePath)

    nNumberUUIDMap = {}
    nNumberDiagMap = {}


    for index, row in table.iterrows():
        nNumber = row['txt_PATHOLOGIE_NUMMER']
       
        uuid = row['uuid']

        diagnosis = str(row['max_class'])

        if str(uuid) == 'nan':
            continue

        if nNumber.startswith("M"):
            continue

        if "glioblastoma" in diagnosis:
            diagnosisChar = "GBM"
        elif "astrocytoma" in diagnosis:
            diagnosisChar = "A"
        elif "oligodendroglioma" in diagnosis:
            diagnosisChar = "O"
        
        nNumber = nNumber.split(" ")[0]
        nNumber = nNumber.strip(",")
        nNumberDiagMap[nNumber] = diagnosisChar
        if not nNumber in nNumberUUIDMap.keys():
            uuids = []
            uuids.append(uuid)
            nNumberUUIDMap[nNumber] = uuids
            nNumberDiagMap[nNumber] = diagnosisChar
        else:
            nNumberUUIDMap[nNumber].append(uuid)
    
    return nNumberUUIDMap, nNumberDiagMap


                        
def collectCaseNumbersFound(path):
    caseNumbersFound = []
    
    for folder_path, folders, files in os.walk(path):
        for slide in files:
            nNumber = extractNNumberFromSlide(slide)
            if not nNumber in caseNumbersFound:
                caseNumbersFound.append(nNumber)
    return caseNumbersFound



def findMissingCases(slidePath, tablePath):

    caseNumbersFound = collectCaseNumbersFound(slidePath)
    caseNumbersInTable, caseDiagMap = collectCaseNumbersInTable(tablePath)

   
    diff = list(set(caseNumbersInTable.keys()) - set(caseNumbersFound))

    for n in caseNumbersFound:
        with open(r'C:\Users\felix\Desktop\dataSetInfo\found.txt', 'a') as found:
                found.write(n)
                found.write('\n')
        found.close()
    for m in caseNumbersInTable.keys():
        with open(r'C:\Users\felix\Desktop\dataSetInfo\inTable.txt', 'a') as inTable:
            for uuid in caseNumbersInTable[m]:
                inTable.write(m +"     "  + str(uuid))
                inTable.write('\n')
        inTable.close()
    for d in diff:
        with open(r'C:\Users\felix\Desktop\dataSetInfo\diff.txt', 'a') as diff:
            diff.write("-------------------------------------------------------------------")
            diff.write('\n')
            uuid = caseNumbersInTable[d]
            diff.write(d+ "  ;  " +  str(uuid)+  "  ;  "+str(caseDiagMap[d]))
            diff.write('\n')
        diff.close()



def cloneSlidesFromFile(pathToFile, outPath, dbPath):
    file = open(pathToFile, 'r')
    lines = file.readlines()

    counter = 0

    for line in lines:

        if line.startswith("---"):
            continue

        split = line.split(";")

        nNumber = split[0]

        uuid= split[1]

        diag = split[2]


        dbFileName = os.path.join(dbPath, uuid+".svs")

        safeFileName = os.path.join(outPath, diag + "-" + nNumber + "-" + str(counter) + ".svs")


        shutil.copyfile(dbFileName,safeFileName)

        counter +=1



def findCaseNumbersWithoutUUID(tablePath):
    return

def findMissingSlidesUUIDs():

    caseNumbers = []

    counterIn =0
    counterOut =0

    for slide in os.listdir(pathOut):
        caseNumber = extractNNumberFromSlide(slide)
        caseNumbers.append(caseNumber)
    for index, row in table1.iterrows():
        nNumber = row['txt_PATHOLOGIE_NUMMER']
        uuid = row['uuid']
        if index < 1310:      
            continue
        if str(uuid) == 'nan':
            continue
         
        if nNumber in caseNumbers:
            counterIn+=1
            continue
        else:
            counterOut+=1
            print(nNumber + "     " + str(uuid))
            with open(r'C:\Users\felix\Desktop\uuids.txt', 'a') as f:
                f.write(nNumber + "     " + str(uuid))
                f.write('\n')

    print("existing slides : " + str(counterIn))
    print("missing slides : " + str(counterOut))


def removeall_inplace(x, l):
    for _ in range(l.count(x)):
        l.remove(x)


def findSlidesWithUUIDAndMissingSVSFile(pathToFile,pathToDB):

    table = pd.read_excel(pathToFile)

    uuidButNoSlides={}
    
    noSlideButOberview = {}

    allFiles = os.listdir(pathToDB)

    for index, row in table.iterrows():


        nNumber = row['txt_PATHOLOGIE_NUMMER']

        uuidCol = str(row['uuid'])

        diagnosis = row['max_class']

        if uuidCol == 'nan':
            continue

        uuids = uuidCol.split(",")

        for uuid in uuids:
            uuid = uuid.replace(" ","")
            slideName = uuid+".svs"
            slidePath = os.path.join(pathToDB,slideName)
            overViewFileName = uuid+"_overview.png"
            overViewFilePath = os.path.join(pathToDB,overViewFileName)

            if not slideName in allFiles:

                if not nNumber in uuidButNoSlides:
                    slides = []

                    slides.append(uuid)

                    uuidButNoSlides[nNumber] = slides
                else:
                    uuidButNoSlides[nNumber].append(uuid)


                if overViewFileName in allFiles:

                    if not nNumber in noSlideButOberview:
                        slides2 = []
                    
                        slides2.append(uuid)

                        noSlideButOberview[nNumber] = slides2
                    else:
                        noSlideButOberview[nNumber].append(uuid)
       
    for u in uuidButNoSlides:
       
        with open(r'/media/np-keller/INTENSO/dataSetInfo/uuidsButNoSlide.txt', 'a') as uuidsButNoSlide:
                uuidsButNoSlide.write("-----------------")
                uuidsButNoSlide.write('\n')
                uuidsButNoSlide.write(str(u) + "   " + str(uuidButNoSlides[u]))
                uuidsButNoSlide.write('\n')
        uuidsButNoSlide.close()

    for n in noSlideButOberview:
        with open(r'/media/np-keller/INTENSO/dataSetInfo/svsMissing.txt', 'a') as svsMissing:
                svsMissing.write("-----------------")
                svsMissing.write('\n')
                svsMissing.write(str(n) + "   " + str(noSlideButOberview[n]))
                svsMissing.write('\n')
        svsMissing.close()

            
    return





def obtainMissingSlides(uuidList):

    file = open('myfile.txt', 'r')
    lines = file.readlines()

    for line in lines:

        uuid = line.split(";")[1].strip()
        name = line.split(";")[0].strip()
        name = name.replace(" ", "_")
        slideName = str(uuid)+".svs"
        slidePath = os.path.join(pathToDB,slideName)

        if not os.path.isfile(slidePath):
            
            continue

        slideDestName = name+".svs"
        slideDestPath = os.path.join(pathOut,slideDestName)

        if slideName in os.listdir(pathToDB):
            print(str(uuid))
        
            shutil.copyfile(slidePath,slideDestPath)
    

       
def cloneSlidesFromDiffFile(pathToFile, outPath, dbPath):
    
    file = open(pathToFile, 'r')
    lines = file.readlines()

    counter = 0

    for line in lines:
        if line.startswith("---"):
            continue

        split = line.split(";")

        nNumber = split[0]
        nNumber = nNumber.replace(" ","")

        uuids= split[1]

        diag = split[2]
        diag = diag.replace(" ","")
        diag = diag.rstrip()

        uuids = uuids.strip(" ")
        uuids = uuids.strip("[")
        uuids = uuids.strip("]")
        uuids = uuids.strip("'")
        uuids = uuids.split(",")

        for uuid in uuids:
            
            uuid = uuid.replace("'","")
            uuid = uuid.replace(" ","")
            if uuid == 'nan':
                continue


            dbFileName = os.path.join(dbPath, uuid+".svs")

            print(dbFileName)

            if os.path.exists(dbFileName):

                fileName = diag + "-" + nNumber + "-" + str(counter) + ".svs"

                safeFileName = os.path.join(outPath, fileName)

                print(safeFileName)

                shutil.copyfile(dbFileName,safeFileName)
           
            counter +=1
           
    return



if __name__ == '__main__':
   #findMissingCases(r"E:\split", r"C:\Users\felix\Downloads\data_Frischgewebe_methylation.xlsx")
   
   
   
   cloneSlidesFromDiffFile(r"/media/np-kirsten/INTENSO/dataSetInfo/diff.txt",r"/media/np-kirsten/INTENSO/a",r"/mnt/NAS4/aperio/data/")
   #cloneSlidesFromDiffFile(r"D:\dataSetInfo\diff.txt",r"D:\a",r"/mnt/NAS4/aperio/data/")

   #findSlidesWithUUIDAndMissingSVSFile(r"/media/np-dennis/INTENSO/dataSetInfo/data_Frischgewebe_methylation.xlsx",r"/mnt/NAS4/aperio/data/")


