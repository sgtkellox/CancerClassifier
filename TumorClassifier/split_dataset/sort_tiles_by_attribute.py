import os
import argparse

def getGroupByLabel(tile, label):
    
    if label == "diag":
        if tile.startswith("A"):
            return "Astro"
        elif tile.startswith("GBM"):
            return "GBM"
        elif tile.startswith("O"):
            return "Oligo"
        
    elif label == "grade":
        diagPart = tile.split("-")[0]
        if diagPart.startswith("GBM"):
            return "four"
        elif diagPart[-1] == str(2):
            return "two"
        elif diagPart[-1] == str(3):
            return "three"
        elif diagPart[-1] == str(4):
            return "four"
        
    elif label == "gradeR":
        diagPart = tile.split("-")[0]
        if diagPart.startswith("GBM"):
            return "high"
        elif diagPart[-1] == str(2):
            return "low"
        elif diagPart[-1] == str(3):
            return "high"
        elif diagPart[-1] == str(4):
            return "high"
            
    elif label == "diff": 
        diagPart = tile.split("-")[0]
        if tile.startswith("A"):
            return "astros"
        elif tile.startswith("GBM"):
            return "astros"
        elif tile.startswith("O"):
            return "oligos"
        
    elif label == "idh":
        diagPart = tile.split("-")[0]
        if tile.startswith("GBM"):
            return "wild"
        else:
            return "mutated"
    



            
        
        
        
    

def sortByDiagnosis(wsis):
    astros = []
    gbms = []
    oligos = []
    
    result = []
    
    for slide in wsis:
        if slide.startswith("A"):
            astros.append(slide)
        elif slide.startswith("GBM"):
            gbms.append(slide)
        elif slide.startswith("O"):
            oligos.append(slide)
            
    result.append(astros)
    result.append(gbms)
    result.append(oligos)
    
    return result


def sortByGrade(wsis):
    
    two = []
    three = []
    four = []
    
    result = []
    
    for slide in wsis:
        diagPart = slide.split("-")[0]
        if diagPart.startswith("GBM"):
            four.append(slide)
        elif diagPart[-1] == str(2):
            two.append(slide)
        elif diagPart[-1] == str(3):
            three.append(slide)
        elif diagPart[-1] == str(4):
            four.append(slide)
            
    result.append(two)
    result.append(three)
    result.append(four)
    
    return result

def sortByRoughGrade(wsis): 
    
    high = []
    low = []
    
    
    result = []
    
    for slide in wsis:
        diagPart = slide.split("-")[0]
        if diagPart.startswith("GBM"):
            high.append(slide)
        elif diagPart[-1] == str(2):
            low.append(slide)
        elif diagPart[-1] == str(3):
            high.append(slide)
        elif diagPart[-1] == str(4):
            high.append(slide)
            
    result.append(high)
    result.append(low)
    
    return result
    


def sortByDifferentiation(wsis):
    
    astros = []
    oligos = []
    
    result = []
    
    for slide in wsis:
        diagPart = slide.split("-")[0]
        if slide.startswith("A"):
            astros.append(slide)
        elif slide.startswith("GBM"):
            astros.append(slide)
        elif slide.startswith("O"):
            oligos.append(slide)
        
            
    result.append(astros)
    result.append(oligos)
    
    return result
    
    
def sortByIDHstatus(wsis):
    mutated = []
    wild = []
    
    result = []
    
    for slide in wsis:
        diagPart = slide.split("-")[0]
        if slide.startswith("A"):
            mutated.append(slide)
        elif slide.startswith("GBM"):
            wild.append(slide)
        elif slide.startswith("O"):
            mutated.append(slide)
                
    result.append(mutated)
    result.append(wild)
    
    return result
    

def printResult(result):
    for entry in result:
        print("--------------")
        for file in entry:
            print(file)
            

 



if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--inPath", help="The path to the folder containing the slides split")
    argParser.add_argument("-o", "--outPath", help="The path to where the new folder is supposed to go")
    argParser.add_argument("-a", "--attribute", help="The path to the folder containing the slides split")
    
    args = argParser.parse_args()

    attr = args.attribute
    
    inPath = args.inPath
    
    outPath = args.outPath
    
    if attr == "diag":
        result = sortByDiagnosis(inPath)
        
    elif attr == "idh":
        result = sortByIDHstatus(inPath)
        
    elif attr == "diff":
        result = sortByDifferentiation(inPath)
    
    elif attr == "grade":
        result = sortByGrade(inPath)
    
    elif attr == "gradeR":
        result = sortByRoughGrade(inPath)
        

    printResult(result)