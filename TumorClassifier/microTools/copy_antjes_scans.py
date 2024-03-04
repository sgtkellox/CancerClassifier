import os
from re import S
import shutil


import pandas as pd

import math
from openpyxl import load_workbook

import argparse



def copySlides(path, outPath,dbPath):
    
    copiedPath = os.path.join(outPath,"copied.txt")
    if os.path.isfile(copiedPath):
        with open(copiedPath) as copiedFile:
            allreadyCopied = [line.rstrip() for line in copiedFile]
    else:
        allreadyCopied = []
         
    table = pd.read_excel(path)
    logPath = os.path.join(outPath,"notFound.txt")
    counter = 0
    for index, row in table.iterrows():
        
        slideName = ""
        if not ("KI-Projekt" in row["else"] or "Ki-Projekt" in row["else"]):
            continue
        if (row["patho-nr"] != "" or row["patho-nr"]!= None ):
            slideName = slideName + str(row["patho-nr"]) 
        else:
            slideName = slideName+"nNumber" 
        
        uuid  = row["uuid"]
        
        if uuid in allreadyCopied:
            print("file with uuid " + uuid + " is allready copied")
            continue
            
        if not (row["preparation"] == "" or row["preparation"]== None ):
            if row["preparation"] == "HE":
                prep = "K"
            elif row["preparation"] == "QUETSCH":
                prep = "Q"
            elif row["preparation"] == "TUPF":
                prep = "T"
            else:
                prep = "wrong"
        else:
            prep = "unreadable"
            
        fileName = slideName + "-"+prep +".svs"
        
        safePath = os.path.join(outPath,fileName)
        
        if os.path.isfile(safePath):
            name = slideName + "-"+prep+str(counter) +".svs"
            safePath = os.path.join(outPath,name)
        
          
         
        filePath = os.path.join(dbPath,uuid+".svs") 
        
        print(filePath)
        
        counter+=1
        
        
        if os.path.isfile(filePath):
            shutil.copy(filePath,safePath)
            with open(copiedPath,"a") as copiedFile:
                copiedFile.write(uuid)
                copiedFile.write('\n')
        else:
            with open(logPath, 'a') as doc:
                doc.write(filePath)
                doc.write('\n')
            
                
        
        print("----------")
        print(safePath)
        

def copyFullyAnnotatedSlides():
    return 

def copyScannedSlides(path,outPath,dbPath):
    copiedPath = os.path.join(outPath,"copied.txt")
    if os.path.isfile(copiedPath):
        with open(copiedPath) as copiedFile:
            allreadyCopied = [line.rstrip() for line in copiedFile]
    else:
        allreadyCopied = []
        
    table = pd.read_excel(path)
    logPath = os.path.join(outPath,"notFound.txt")
    counter = 0
       
        
    for index, row in table.iterrows():
         
        slideName = ""
    
        if row[0] == "PXA":
            slideName = slideName +"PXA-"
        elif row[0] == "Pilocytic astrocytoma":
            slideName = slideName +"PA-"     
        elif row[0] == "Metastasis":
            slideName = slideName +"MET-"    
        elif row[0] == "Medulloblastoma":
            slideName = slideName +"MB-"
        elif row[0] == "Lymphoma":
            slideName = slideName +"LYM-"
        elif row[0] == "Ependymoma":
            slideName = slideName +"LYM-"
        
        slideName = slideName + str(row[1])
        slideName = slideName + "-" + str(counter)+".svs"
    
        slideOrig = os.path.join(dbPath, str(row[2])+".svs")
        
        safePath = os.path.join(outPath,slideName)
        
        counter +=1
        
        print(slideName)
        
        
        if os.path.isfile(slideOrig):
            shutil.copy(slideOrig,safePath)
            with open(copiedPath,"a") as copiedFile:
                copiedFile.write(slideOrig)
                copiedFile.write('\n')
        else:
            with open(logPath, 'a') as doc:
                doc.write(slideOrig)
                doc.write('\n')
        
        
    
    
          
        
if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    
    argParser.add_argument("-t", "--tables", help="The path to the table")
    argParser.add_argument("-p", "--outPath", help="The path to where the files are supposed to go")
    argParser.add_argument("-o", "--origPath", help="The NAS path where the slides are stored")
    
    args = argParser.parse_args()

    tables = args.tables
    outpath = args.outPath
    origPath = args.origPath
    
   
            
          
            
    
        
   
    
            
    copySlides(tables,outpath,origPath)
             
        
    
    
    
    
    
    
    
    
    
    
    