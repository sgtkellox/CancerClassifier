def calcPixelPosition(image):
    #print(image)
    splitP1 = image.split("_")
    x = int(int(splitP1[1])/512)
    y = int(int(splitP1[2].split(".")[0])/512)
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

