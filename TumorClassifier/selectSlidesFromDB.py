import os
import shutil

import pandas as pd

import math
#from openpyxl import load_workbook


def copySlides():
    for index, row in table1.iterrows():

        if index < 1310:
            
            continue 

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


def checkForMissingFiles():

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

       


    

        



        



if __name__ == '__main__':
    
    pathTable1 = r"E:\script\DG.xlsx"

    pathToDB = r"E:\uuids"
    pathOut = r"E:\copyTest"

    table1 = pd.read_excel(pathTable1)

    checkForMissingFiles()
