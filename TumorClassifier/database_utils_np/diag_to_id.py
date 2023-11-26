import os 
from openpyxl import load_workbook
import pandas as pd

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
        if row["Patho-Nr"] == nNumber:
            app = "fail"
            diag = row["Diagnosis"]
            
            if diag == "PXA":
                app = "PXA"
            elif diag == "Pilocytic astrocytoma":
                app = "PA"     
            elif diag == "Metastasis":
                app = "MET"    
            elif diag == "Medulloblastoma":
                app = "MB"
            elif diag == "Lymphoma":
                app = "LYM"
            elif diag == "Ependymoma":
                app = "LYM"
            elif diag == "Neurinom":
                app = "N"
            elif diag == "Hypophysenadenom":
                app = "PIT"
            elif diag == "H3-mutiertes Gliom":
                app = "H3"
            elif diag == "Meningeoma":
                app = "MEN"
            
            name = app +"-" +nNumber
            name = name + "-" +prep
            print(name)
            return name
            
    return app
            
            
    



def processFolder(path, table):
    for wsi in os.listdir(path):
        notFound = 
        if wsi.endswith(".svs"):
            
            fullName = lookUp(table,wsi)

            

path = r"D:\new entities\Kryo"
tablePath = r"C:\Users\felix\Downloads\All_New_Cases_SS_11_entities(1).xlsx"



if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    
    argParser.add_argument("-p", "--path", help="")
    argParser.add_argument("-t", "--tablePath", help="The path to the table")
    
    
    args = argParser.parse_args()

    tablePath= args.tablePath
    path = args.path

    table = pd.read_excel(tablePath)

    processFolder(path,table) 
   
            
            

