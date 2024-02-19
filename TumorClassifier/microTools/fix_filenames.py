import argparse
import os


if __name__ == '__main__':



    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--inPath", help="The path to the folder containing the slides")
    
    args = argParser.parse_args()

    inPath = args.inPath
    
    for slide in os.listdir(inPath):
        if slide.endswith(".svs"):
            fileName = slide.split(".")[0]
            split = fileName.split("-")
            newName = split[0]+"-"+split[1]+"-"+split[2]+"-"+split[3]+".svs"
            newPath = os.path.join(inPath, newName)
            oldPath = os.path.join(inPath, slide)
            
    
    