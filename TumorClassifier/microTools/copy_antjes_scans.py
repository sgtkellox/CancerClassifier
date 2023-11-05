import os
import shutil

import pandas as pd

import math
from openpyxl import load_workbook

import argparse



def copySlides(path, outPath):
    table = pd.read_excel(path)
    counter = 0
    for index, row in table.iterrows():
        slideName = ""
        if not ("KI-Projekt" in row["original_file_name"] or "Ki-projekt" in row["original_file_name"]):
            continue
        if (row["patho_id"] != "" or row["patho_id"]!= None ):
            slideName = slideName + str(row["patho_id"])
        else:
            slideName = slideName+"nNumber" + str(counter)
        
        if not (row["staining_type"] != "" or row["staining_type"]!= None ):
            if row["staining_type"] == "HE":
                prep = "K"
            elif row["staining_type"] == "Quetsch":
                prep = "Q"
            elif row["staining_type"] == "Tupf":
                prep = "T"
        else:
            prep = "unreadable"
            
        slideName = slideName + "-"+prep +".svs"
        
        safePath = os.path.join(outPath,slideName)
         
        filePath = row["file_location"] + row["uuid"]+ ".svs"
        
        counter+=1
        
        logPath = os.path.join(outPath,"notFound.txt")
        
        if os.path.isfile(filePath):
            shutil.copy(filePath,safePath)
        else:
            with open(logPath, 'a') as doc:
                doc.write(filePath)
                doc.write('\n')
        
        print("----------")
        print(outPath)
        print(filePath)
        
        
        
        
                
            
        
            
            
            
            
            
        
            
            
        
    






if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    
    argParser.add_argument("-t", "--table", help="The path to the table")
    argParser.add_argument("-p", "--outPath", help="The path to where the files are supposed to go")
    
    args = argParser.parse_args()

    table = args.table
    outpath = args.outPath
    
    copySlides(table,outpath)
    
    
    
    
    
    
    
    
    
    
    