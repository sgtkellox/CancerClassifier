import os
import argparse
import math


import numpy as np

import torch
import cv2
import torchvision.transforms as transforms

from tqdm import tqdm


from tile_creation.tile_utils import calcPixelPosition

#from tile_creation.filter_utils import open_slide

from modell_training.binary_classifier.model import CustomCNN


add_dll_dir = getattr(os, "add_dll_directory", None)
vipsbin = r"C:\AI\vips-dev-8.14\bin"
if callable(add_dll_dir):
    add_dll_dir(vipsbin)
    print("added dir")
else:
    os.environ["PATH"] = os.pathsep.join((vipsbin, os.environ["PATH"]))
    print("added path")

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


    

import sys
import random
import pyvips

labels = ['Artifact', 'HQ-Tissue']



def sortTilesByWSI(path):

    wsis = {}

    for img in os.listdir(path):

        wsiName = img.split("_")[0]

        if wsiName in wsis:
            wsis[wsiName].append(img)
        else:
            wsis[wsiName] = []
            wsis[wsiName].append(img)
    print("finished sorting by wsi")
    return wsis


def extractTileCoordinates(image):

    splitP1 = image.split("_")
    x = int(splitP1[1])
    y = int(splitP1[2].split(".")[0])
    return x , y

def extractXCoordinate(tile):
    splitP1 = tile.split("_")
    x = int(splitP1[1])
    return x




def getWsiDimensions(nNumber, slidePath):
    slides = os.listdir(slidePath)
   
    for wsi in slides:
        wsiNnumber = wsi.split(".")[0]

        split = wsiNnumber.split("-")

        wsiNnumber = split[1] + "-"  + split[2] + "-" + split[3]
        
        split2 = nNumber.split("-")

        print("tileNumber :" + nNumber)

        tileNNumber = split2[1] + "-"  + split2[2] + "-" + split2[3]
        
        
        if wsiNnumber == tileNNumber:
               
            slidePath = os.path.join(slidePath,wsi)
            slide = open_slide(slidePath)
            a = slide.dimensions
            return a[0] , a[1]
                    
    return 0, 0


def findWidhHeight(images):
    minX = 100000
    maxX = 0
    minY = 100000
    maxY = 0
    for image in images:
   
        x,y = extractTileCoordinates(image)

        minX = min(minX,x)
        maxX = max(maxX,x)
        minY = min(minY,y)
        maxY = max(maxY,y)

    
    if minX == 0:
        xshift = 0
    else:
        xshift = minX-tileSize
    if minY == 0:
        yShift = 0
    else:
        yShift = minY-tileSize

    
    width = maxX+1000-minX
    height = maxY + 1000 - minY

    

    print("width " +str(width))
    print("height " +str(height))

    print("maxX " +str(maxX-xshift))
    print("maxY " +str(maxY-yShift))

    return width, height , xshift, yShift


def sortByXCoordinate(images):

    xCoords = []

    for image in images:
        xCoords.append(extractXCoordinate(image))

    xCoords = list(set(xCoords))
    
    xCoords.sort()
         
    return xCoords


def findGaps(xCoords, tileSize, k):
    gaps = []
    for i in range(0,len(xCoords)-1):
        if xCoords[i+1] - xCoords[i] >= k * tileSize:
            gap = (xCoords[i], xCoords[i+1])
            gaps.append(gap)
    print(gaps)
    return gaps

def mergeGaps(gaps, tileSize, k):
    
    for index, gap in enumerate(gaps.copy()[0:-1]):               
        if gaps[index+1][0] - gap[1] <= k *tileSize:
            mergedGap = (gap[0], gaps[index+1][1])
            del gaps[index+1]
            del gaps[index]          
            gaps.insert(index,mergedGap)
            mergeGaps(gaps,tileSize,k)
            break

    return gaps

def getGapLenght(gap):
    return gap[1] - gap[0]


def addUpGaps(gaps, index, tileSize = 512):
    gapSum = 0
    for i in range(index):
        
        gapSum = gapSum + getGapLenght(gaps[i]) - tileSize
    return gapSum

def adjustTileCoords(tile, gaps, tileSize):
    x = extractXCoordinate(tile)
    if x > gaps[-1][1]:
        newX = x-addUpGaps(gaps,len(gaps),tileSize)
        return newX
    elif x < gaps[0][0]:
        return x
    for i in range(len(gaps)-2):
        if x > gaps[i][1] and x < gaps[i+1][0]:
            newX = x-addUpGaps(gaps,i,tileSize)
            return          
    return

def adjustXDim(x, gaps, tileSize):
    return x - addUpGaps(gaps, len(gaps), tileSize)
     
    
def makeResultFolder(outPath):

    aPath = os.path.join(outPath,"artefact")
    if not os.path.isdir(aPath):
        os.mkdir(aPath)

    gPath = os.path.join(outPath,"good")
    if not os.path.isdir(gPath):
        os.mkdir(gPath)

    diffPath = os.path.join(outPath,"difficult")
    if not os.path.isdir(diffPath):
        os.mkdir(diffPath)

    mapPath = os.path.join(outPath,"maps")
    if not os.path.isdir(mapPath):
        os.mkdir(mapPath)

    wsiPath  = os.path.join(outPath,"wsis")
    if not os.path.isdir(wsiPath):
        os.mkdir(wsiPath)



    return aPath,gPath,diffPath,mapPath,wsiPath


def makeTileMap(tilePath, imgs, outPath ,slideWidth, slideHeight, xshift,yshift, model, transform,tileSize, gaps):


    result = pyvips.Image.black(slideWidth+tileSize,slideHeight,bands=3)

    

    slideWidth = int(slideWidth/tileSize)
    slideHeight = int(slideHeight/tileSize)

    tileMap = np.zeros((slideHeight+1, slideWidth+20, 1), np.uint8)

    


    aPath = os.path.join(outPath,"artefact")
    gPath = os.path.join(outPath,"good")
    diffPath = os.path.join(outPath,"difficult")
   

    for img in tqdm(imgs):   

        if ".ini" in img:
            continue
   
        x,y = calcPixelPosition(img,xshift,yshift,tileSize)
        
        x = adjustTileCoords(img, gaps,tileSize)
   
        imgFullPath = os.path.join(tilePath,img)


        tile = pyvips.Image.new_from_file(imgFullPath, access='sequential')

        image = cv2.imread(imgFullPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_image = image.copy()
        image = transform(image)
       
        image = torch.unsqueeze(image, 0)
        image = image.to(device)


        # Forward pass throught the image.
        outputs = model(image)
        output_sigmoid = torch.sigmoid(outputs)
        pred_class = 1 if output_sigmoid > 0.5 else 0


        if pred_class==1:
            safePath = os.path.join(aPath,img)
            tile = tile.new_from_image(250).bandjoin(tile[1:3])
            
        else:
            safePath = os.path.join(gPath,img)
           
        tileMap[y][x]= pred_class+1

       
            

        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)


        absX,absY = extractTileCoordinates(img)

        

        absX = absX-xshift
        absY = absY -yshift

       

        result = result.insert(tile,absX,absY)
     
        #cv2.imwrite(safePath,orig_image)    
        
        
               
    return tileMap, result

def drawResultImage(resultsArray, slideWidth, slideHeight):

     slideWidth = int(slideWidth/tileSize)
     slideHeight = int(slideHeight/tileSize)
     result = np.zeros((slideHeight*10, slideWidth*10, 3), np.uint8)
     result.fill(255)

     numArt=0
     numGood = 0

     for x in range(len(resultsArray)):
         for y in range(len(resultsArray[0])):
             
             if resultsArray[x][y]==1:
                 numArt +=1
                 result[x*10:x*10+10,y*10:y*10+10] = [255, 0, 0]
             elif resultsArray[x][y]==2:
                 numGood +=1
                 result[x*10:x*10+10,y*10:y*10+10] = [0, 255, 0]
             

     cv2.putText(
        result, "Artefact " + str(numArt),
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255, 0, 0), 2, lineType=cv2.LINE_AA
     )
    # Annotate the image with prediction.
     cv2.putText(
        result, "HQ-Tissue " + str(numGood),
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA
     )
     
    
     return result
    

def printWsis(wsis):

    for slide in wsis:
        print(slide+str(len(wsis[slide])))



def makeClassificationRun(tilePath, outPath, model, transform,tileSize):
    wsis = sortTilesByWSI(tilePath)

    aPath, gPath , diffPath, mapPath,wsiPath  = makeResultFolder(outPath)

    printWsis(wsis)
    
    for slide in tqdm(wsis):
        
        print("processing slide"+ slide)

        xCoords = sortByXCoordinate(wsis[slide])

        gaps = findGaps(xCoords, tileSize, 3)
        

        slideWidth , slideHeight, xshift, yshift = findWidhHeight(wsis[slide])
        
        slideWidth = adjustXDim(slideWidth, gaps, tileSize)

        #print("dims of slide " + slide + " with dimensions w: " + str(slideWidth) +" and "+ str(slideHeight))

        w = int(slideWidth/tileSize)
        h = int(slideHeight/tileSize)

        print("dims of out image w: " + str(w)+ "h: " + str(h))

        if slideWidth == 0 or slideHeight == 0:
           print("Warning: the slide "+ slide +" has dims 0 , 0")
           continue
       
       

        tileMap, result = makeTileMap(tilePath,wsis[slide],outPath,slideWidth, slideHeight,xshift,yshift, model, transform,tileSize,gaps)

        

        tifPath = os.path.join(wsiPath,slide+".tif")
    
       
       
        resultImg = drawResultImage(tileMap,slideWidth, slideHeight)

        safePath = os.path.join(mapPath,slide+".jpg")

        resultImg = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)  

        cv2.imwrite(safePath, resultImg)

        print("stitching result")

        result.tiffsave(tifPath, compression='jpeg', 
                  tile=True, tile_width=512, tile_height=512, 
                  pyramid=True,  bigtiff=True)

        print("finished")

        


       

if __name__ == '__main__':


     tilePath = r"C:\Users\felix\Desktop\stitcherTestIn"

     slidePath =r"D:\slides\kryoQ1"

     outPath = r"C:\Users\felix\Desktop\stitcherTestOut"

     tileSize = 512


     

     device = ('cuda' if torch.cuda.is_available() else 'cpu')

     transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
    ])

     model = CustomCNN(num_classes=1)
     checkpoint = torch.load(r'C:\Users\felix\Desktop\AutoEncoder\models2\110.pth', map_location=device)
     print('Loading trained model weights...')
     model.load_state_dict(checkpoint['model_state_dict'])
     
     
     model = model.to(device)

     makeClassificationRun(tilePath, outPath, model, transform,tileSize)