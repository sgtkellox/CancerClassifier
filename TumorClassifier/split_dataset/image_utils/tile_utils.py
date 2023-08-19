


def calcPixelPosition(image):
    splitP1 = image.split("_")
    x = int(splitP1[1])/500
    y = int(splitP1[2].split(".")[0])/500
    return x , y 


def calcGapBetweenTiles(tileA,tileB):
    posAX, posAY = calcPixelPosition(tileA)
    posBX, posBY = calcPixelPosition(tileB)

    gapX = posAX-posBX
    gapY = posBY-posAY

    return gapX, gapY



def extractNNumberFromSlide(slide):

    parts = slide.split("-")
    nNumber = parts[1] + "-" + parts[2]

    return nNumber


def extractIndetifierFromSlide(slide):

    parts = slide.split("-")
    identifier = parts[1] + "-" + parts[2] + "-" +parts[3]

    return identifier

def getPreparation(slide):
    identifier = slide.split("_")[0]
    prep = slide.split("-")[3]
    if prep[-1].isdigit():
        prep = prep[0]
    if prep =="K":
        return "kryo"
    elif prep =="Q":
        return "smear"
    else:
        return "touch"

def getDiagnosis(slide):
     diag = slide.split("-")[0]
     if "A" in diag:
         return "Astro";
     elif "GBM" in diag:
         return "GBM";
     else:
         return "O"


    

