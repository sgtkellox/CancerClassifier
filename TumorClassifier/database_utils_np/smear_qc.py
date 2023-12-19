import os

def extractNNumberFromWsi(wsi):
    wsi = wsi.split(".")[0]
    split = wsi.split("-")
    nNumber = split[1]+"-"+split[2]
    return nNumber


def extraxtPrepInfo(wsi):
    wsi = wsi.split(".")[0]
    split = wsi.split("-")
    prep = split[3]
    if prep[-1].isdigit():
        return prep[0]
    else: 
        return prep
    

def findBadQualitySlides(slide):
    split = slide.split("-")
    if len(split)>4:       
        quality = split[4].split(".")[0]
        if quality == "B" or quality=="Q0":
           
            return True
        else:
            return False
    else:
        return False
        
        
            

def filterData(path):
    badSlides = []
    for folder in os.listdir(path):
        folderPath = os.path.join(path,folder)
        
        for slide in os.listdir(folderPath):
            
            if slide.endswith(".svs"):      
                
                if extraxtPrepInfo(slide) == "Q":
                    
                
                    if findBadQualitySlides(slide)==True:
                        
                        nNumber = extractNNumberFromWsi(slide)
                    
                        badSlides.append(nNumber)
                    
    return badSlides

def printList(numbers, path):
    with open(path, 'a') as doc:
        for number in numbers:
            doc.write(number)
            doc.write('\n')
    doc.close()
    

if __name__ == '__main__':
    
    inPath = r"D:\slides"
    outPath = r"F:\check\bq.txt"
    
    files = filterData(inPath)
    files.sort()
    printList(files,outPath)
    
    
    
    
    
    

                    
                
                
                    