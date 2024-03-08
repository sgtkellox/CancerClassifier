import os 
from openpyxl import load_workbook
import pandas as pd

import time

import argparse

def extractNNumberFromWsi(wsi):
    wsi = wsi.split(".")[0]
    split = wsi.split("-")
    nNumber = split[0]+"-"+split[1]
    return nNumber

def extraxtPrepInfo(wsi):
    wsi = wsi.split(".")[0]
    split = wsi.split("-")
    prep = split[2]
    if len(split)>3:
        prep = prep+"-"+split[3]
    return prep


def printNotFoundFiles(files, path):
    
    with open(path, 'a') as doc:
        for file in files:
            doc.write(file)
            doc.write('\n')
    doc.close()
    

    


def lookUp(table,wsi):
    nNumber = extractNNumberFromWsi(wsi)
    prep = extraxtPrepInfo(wsi)
    
    for index, row in table.iterrows():
        app = "fail"
        if row["Patho-Nr"] == nNumber:
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
            name = app +"-" +nNumber
            name = name + "-" +prep+".svs"
            return name
            
    return app

def lookUp2(table,wsi):
    nNumber = extractNNumberFromWsi(wsi)
    prep = extraxtPrepInfo(wsi)
    
    for index, row in table.iterrows():
        app = "fail"
        if row["Patho-Nr."] == nNumber:
            diag = row["Entity"]
            app = diag
           
            name = app +"-" +nNumber
            name = name + "-" +prep+".svs"
            return name
            
    return app


def lookUpWithUUid(uuid, table, counter):
    
    
    for index, row in table.iterrows():
        app = "fail"
        #print(uuid)
        if row[2] == uuid:
            diag = row[0]
            nNumber = row[1]
            
            
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
            name = app +"-" +nNumber
            name = name + "-" +str(counter)+".svs"
            
            return name
    return app
    


def correctName(path):
    for slide in os.listdir(path):
        split = slide.split("-")
        if split[0] == "fail" or split[0] == "N" or split[0]=="LYM":
            newName = split[1]+"-"+split[2]+"-"+split[3]
            if len(split)>4:
                
                newName = newName + "-" + split[4]
            oldPath = os.path.join(path,slide)
            newPath = os.path.join(path,newName)
            os.rename(oldPath,newPath)
                
            
            
            
            

def processFolder(path, table):
    notFound = []
    for wsi in os.listdir(path):
        if wsi.endswith(".svs"):     
            fullName = lookUp2(table,wsi)
            
            if fullName == "fail":
                notFound.append(wsi)
            else:
                oldPath = os.path.join(path,wsi)
                newPath = os.path.join(path,fullName)
                print(newPath)
                os.rename(oldPath,newPath)
    if len(notFound)!=0:
        notFoundPath = os.path.join(path,"notFound.txt")
        printNotFoundFiles(notFound,notFoundPath)
    
              
def asignFileNameFromUUID(slidePath, table):
    
    counter = 0
    for slide in os.listdir(slidePath):
        if not slide.endswith(".svs"):
            continue
        fileUUID = slide.split(".")[0]
        name = lookUpWithUUid(fileUUID, table,counter)
        counter +=1
        
        oldPath = os.path.join(slidePath,slide)
        newPath = os.path.join(slidePath,name)
        os.rename(oldPath,newPath)
        
        
        
        

            





if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    
    argParser.add_argument("-p", "--path", help="")
    argParser.add_argument("-t", "--tablePath", help="The path to the table")
    
    
    args = argParser.parse_args()

    tablePath= args.tablePath
    path = args.path

    table = pd.read_excel(tablePath)
    
    

    #correctName(path)
    
    #time.sleep(10)

    processFolder(path,table) 
   
            
            

