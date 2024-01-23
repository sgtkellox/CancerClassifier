import os
import shutil

import pandas as pd

import argparse

import math
from openpyxl import load_workbook



def copyScannedSlides(tablePath,outPath,origPath):
    table = pd.read_excel(tablePath)
    logPath = os.path.join(outPath,"notFound.txt")
    
       
        
    for index, row in table.iterrows():
        fileDestName = row["Patho-Nr"]
        uuid = row["uuid"]
        fileOrigPath = os.path.join(origPath,uuid+".svs")
        fileDestPath = os.path.join(outPath,fileDestName+".svs")
        if os.path.isfile(fileOrigPath):
            shutil.copy(fileOrigPath,fileDestPath)
            with open(logPath,"a") as copiedFile:
                copiedFile.write(uuid)
                copiedFile.write('\n')
        else:
            with open(logPath, 'a') as doc:
                doc.write(uuid)
                doc.write('\n')
        
        




if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    
    argParser.add_argument("-t", "--table", help="The path to the table")
    argParser.add_argument("-p", "--outPath", help="The path to where the files are supposed to go")
    argParser.add_argument("-o", "--origPath", help="The NAS path where the slides are stored")
    
    args = argParser.parse_args()

    table = args.table
    outpath = args.outPath
    origPath = args.origPath
    
          
        
    if table.endswith(".xlsx"):
            
                     
        copyScannedSlides(table,outpath,origPath)