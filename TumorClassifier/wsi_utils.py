


def calcPixelPosition(image):
    splitP1 = image.split("_")
    x = int(int(splitP1[1])/500)
    y = int(int(splitP1[2].split(".")[0].strip("cam"))/500)

    #print("X "+ str(x)+ " Y " + str(y))

    return x , y 

def calcGapBetweenTiles(tileA,tileB):
    posAX, posAY = calcPixelPosition(tileA)
    posBX, posBY = calcPixelPosition(tileB)

    gapX = posAX-posBX
    gapY = posBY-posAY

    return gapX, gapY

def sortTilesByXPosition():
    return



