from genericpath import isfile
import os 

import argparse

import pandas as pd

import math


def getTMAIdent(tma):
    return tma.split("-")[2]



def lookUpNumber(folder, position, table):
    for index, row in table.iterrows():
        if row["TMA.short"]== folder and row["position"]==position:
            print("match folder " + folder  + "  with  " + str(row["TMA.short"]) + " in table")
            print("match file " + position  + "  with  " + str(row["position"]) + " in table")
            grade = row["WHO.2016"]
            if  not math.isnan(grade):                
                grade = str(int(grade))   
            else:
                return "empty"
            if grade.strip() != "":
                return grade
            else:
                return "empty"
 
            

def printMap(coreMaps, outPath):
    with open(outPath, 'w') as doc:
        for coreMap in coreMaps:
            for key, value in coreMap.items():
                doc.write(key+ "   " + value)
                doc.write('\n')
    doc.close()
            

def asignLabels(imagePath, table):
    tmaMaps = []
    for folder in os.listdir(imagePath):
        print("------------------------------")
        folderpath = os.path.join(imagePath,folder)
        if os.path.isfile(folderpath):
            continue
        tmaMap = {}
        tmaIdentifier = getTMAIdent(folder)
        for file in os.listdir(folderpath):
            if file.endswith(".tif"):
                position = file.split(".")[0]
                grade = lookUpNumber(folder, position, table)
                fileIndent = str(tmaIdentifier)+"-"+file.split(".")[0]
                if grade != "empty" and grade != None:
                    tmaMap[fileIndent] = grade
        tmaMaps.append(tmaMap)
    printMap(tmaMaps, os.path.join(imagePath,"label.txt"))
                    
                

        

if __name__ == '__main__':
    
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-c", "--cores", help="The Path to the cores")
    argParser.add_argument("-t", "--table", help="the path to the table")
    
    
    args = argParser.parse_args()

    slides = args.cores
    tablePath = args.table
    
    table = pd.read_excel(tablePath)
    asignLabels(slides, table)
    
    
    
    
    
    


   
    
                
            


    

            
        