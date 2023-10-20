import os


def sortByFileName(inPath, attribute):
    
    for file in os.listdir(inPath):
        
    
        if attribute == "Diagnosis":
            print()
            
    return 


def sortByDiagnosis(inPath):
    astros = []
    gbms = []
    oligos = []
    
    result = []
    
    for slide in os.listdir(inPath):
        if slide.startswith("A"):
            astros.append(slide)
        elif slide.startswith("GBM"):
            gbms.append(slide)
        elif slide.startswith("O"):
            oligos.append(slide)
            
    result.append(astros,gbms,oligos)
    
    return result


def sortByGrade(inPath):
    
    two = []
    three = []
    four = []
    
    result = []
    
    for slide in os.listdir(inPath):
        diagPart = slide.split("-")[0]
        if diagPart.startswith("GBM"):
            four.append(slide)
        elif diagPart[-1] == str(2):
            two.append(slide)
        elif diagPart[-1] == str(3):
            three.append(slide)
        elif diagPart[-1] == str(4):
            four.append(slide)
            
    result.append(two,three,four)
    
    return result

def sortByRoughGrade(inPath): 
    
    high = []
    low = []
    
    
    result = []
    
    for slide in os.listdir(inPath):
        diagPart = slide.split("-")[0]
        if diagPart.startswith("GBM"):
            high.append(slide)
        elif diagPart[-1] == str(2):
            low.append(slide)
        elif diagPart[-1] == str(3):
            high.append(slide)
        elif diagPart[-1] == str(4):
            high.append(slide)
            
    result.append(high,low)
    
    return result
    


def sortByDifferentiation(inPath):
    
    astros = []
    oligos = []
    
    result = []
    
    for slide in os.listdir(inPath):
        diagPart = slide.split("-")[0]
        if slide.startswith("A"):
            astros.append(slide)
        elif slide.startswith("GBM"):
            astros.append(slide)
        elif slide.startswith("O"):
            oligos.append(slide)
        
            
    result.append(astros,oligos)
    
    return result
    
    
def sortByIDHstatus(inPath):
    mutated = []
    wild = []
    
    result = []
    
    for slide in os.listdir(inPath):
        diagPart = slide.split("-")[0]
        if slide.startswith("A"):
            mutated.append(slide)
        elif slide.startswith("GBM"):
            wild.append(slide)
        elif slide.startswith("O"):
            mutated.append(slide)
                
    result.append(mutated,wild)
    
    return result
    
    
        


def sortByLabelFile(inPath, attribute):
    

    return 

    
    
    