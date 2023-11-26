import os
from re import S
import shutil


import pandas as pd

import math
from openpyxl import load_workbook

import argparse



def copySlides(path, outPath):
    
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
        if not ("KI-Projekt" in row["original_file_name"] or "Ki-Projekt" in row["original_file_name"]):
            continue
        if (row["patho_id"] != "" or row["patho_id"]!= None ):
            slideName = slideName + str(row["patho_id"]) +str(counter)
        else:
            slideName = slideName+"nNumber" + str(counter)
        
        uuid  = row["uuid"]
        
        if uuid in allreadyCopied:
            print("file with uuid " + uuid + " is allready copied")
            continue
            
        if not (row["staining_type"] == "" or row["staining_type"]== None ):
            if row["staining_type"] == "HE":
                prep = "K"
            elif row["staining_type"] == "QUETSCH":
                prep = "Q"
            elif row["staining_type"] == "TUPF":
                prep = "T"
            else:
                prep = "wrong"
        else:
            prep = "unreadable"
            
        slideName = slideName + "-"+prep +".svs"
        
        safePath = os.path.join(outPath,slideName)
        
          
         
        filePath = row["file_location"] + uuid+ ".svs"
        
        
        
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
    
    for table in os.listdir(tables):
        print("----------------")
        print("Processing table " + table)
        
        if table.endswith(".xlsx"):
            
            tablePath = os.path.join(tables,table)       
            
            tableName = table.split(".")[0]
        
            tableFolder = os.path.join(outpath, tableName)
            os.mkdir(tableFolder)
            
            copyScannedSlides(tablePath,tableFolder,origPath)
             
        
    
    
    
    
    
    
    
    
    
    
    