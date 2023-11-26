import os 
from openpyxl import load_workbook
import pandas as pd

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
    


def lookUp(table,wsi):
    nNumber = extractNNumberFromWsi(wsi)
    prep = extraxtPrepInfo(wsi)
    
    for index, row in table.iterrows():
        if row["Patho-Nr"] == nNumber:
            diag = row["Diagnosis"]
            
            if diag == "PXA":
                app = "PXA-"
            elif diag == "Pilocytic astrocytoma":
                app = "PA-"     
            elif diag == "Metastasis":
                app = "MET-"    
            elif diag == "Medulloblastoma":
                app = "MB-"
            elif diag == "Lymphoma":
                app = "LYM-"
            elif diag == "Ependymoma":
                app = "LYM-"
            elif diag == "Neurinom":
                app = "N-"
            elif diag == "Hypophysenadenom":
                app = "PIT-"
            elif diag == "H3-mutiertes Gliom":
                app = "H3-"
            name = diag +"-" +nNumber
            name = name + "-" +prep
            print(name)
            return
            
    print("not found " +nNumber)
            
            
    



def processFolder(path, table):
    for wsi in os.listdir(path):
        if wsi.endswith(".svs"):
            
            lookUp(table,wsi)
            

path = r"D:\new entities\Kryo"
tablePath = r"C:\Users\felix\Downloads\All_New_Cases_SS_11_entities(1).xlsx"
table = pd.read_excel(tablePath)

processFolder(path,table)          
            
            

