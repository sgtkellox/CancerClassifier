import os

import pandas as pd

def extractNNumberFromWsi(wsi):
    wsi = wsi.split(".")[0]
    split = wsi.split("-")
    nNumber = split[1]+"-"+split[2]
    return nNumber

def listFromTable(tablePath,safePath):
    table = pd.read_excel(tablePath)
    

    with open(safePath, 'a') as doc:
                
    
        for index, row in table.iterrows():
            
            nNumber = row["Patho-Nr"]
            
            doc.write(nNumber)
            doc.write('\n')
            
def listFromFolder(slidePath,safePath):
    

     with open(safePath, 'a') as doc:

        for slide in os.listdir(slidePath):
            nNumber = extractNNumberFromWsi(slide)
            doc.write(nNumber)
            doc.write('\n')
            

if __name__ == '__main__':
    tablePath = r"C:\Users\felix\Downloads\All_New_Cases_SS_11_entities(1).xlsx"
    filesP1 = r"D:\slides\kryoQ1"
    filesP2 = r"D:\slides\kryoQ2"
    
    safePath = r"C:\Users\felix\Desktop\list.txt"
    
    listFromTable(tablePath,safePath)
    listFromFolder(filesP1,safePath)
    listFromFolder(filesP2,safePath)
        
    




