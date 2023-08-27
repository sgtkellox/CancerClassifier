import os
import shutil

import argparse


def readFile(filePath):
    with open(filePath) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        #print(lines)
        for index, line in enumerate(lines):
            if line == "test":
                print(lines[index+2:])
                return lines[index+2:] 


def selectFileByNNumber(nNumber, slidePath, outPath):
    slides = os.listdir(slidePath)
   
    for wsi in slides:
        wsiNnumber = wsi.split(".")[0]

        split = wsiNnumber.split("-")

        wsiNnumber = split[1] + "-"  + split[2] + "-" + split[3]
        
        split2 = nNumber.split("-")

        

        tileNNumber = split2[1] + "-"  + split2[2] + "-" + split2[3]
        
        
        if wsiNnumber == tileNNumber:
               
            slidePath = os.path.join(slidePath,wsi)
            safePath = os.path.join(outPath,wsi)
            shutil.copy(slidePath,safePath)
 
def selectSlides(slidePath, docPath, safePath):

    slides = readFile(docPath)


    
    for slide in slides:
        selectFileByNNumber(slide,slidePath,safePath)





if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--inPath", help="The Path where ur datasplit is stored")
    argParser.add_argument("-o", "--outPath", help="the path where the outfile is stored")
    argParser.add_argument("-f", "--file", help="the path where the outfile is stored")

    args = argParser.parse_args()

    inPath = args.inPath
    docPath = args.file
    outPath = args.outPath

    inPath = r"F:\slides\kryo"
    docPath = r"C:\Users\felix\Desktop\split_500_sn.txt"
    outPath = r"C:\Users\felix\Desktop\fixedKryoTest\kryoTestSn"

    selectSlides(inPath, docPath,outPath)

    
    
