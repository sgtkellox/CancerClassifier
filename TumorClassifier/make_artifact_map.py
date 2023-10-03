import os
import argparse
import math


import numpy as np

import torch
import cv2
import torchvision.transforms as transforms


from tile_creation.tile_utils import calcPixelPosition

from tqdm import tqdm

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
        xshift = minX-500
    if minY == 0:
        yShift = 0
    else:
        yShift = minY-500

    print("minY " + str(minY))
    width = maxX+500-minX
    height = maxY + 500 - minY

    return width, height , xshift, yShift





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


def makeTileMap(tilePath, imgs, outPath ,slideWidth, slideHeight, xshift,yshift, model, transform):


    result = pyvips.Image.black(slideWidth,slideHeight,bands=3)

    

    slideWidth = int(slideWidth/500)
    slideHeight = int(slideHeight/500)

    tileMap = np.zeros((slideHeight+10, slideWidth+10, 1), np.uint8)

    


    aPath = os.path.join(outPath,"artefact")
    gPath = os.path.join(outPath,"good")
    diffPath = os.path.join(outPath,"difficult")
   

    for img in tqdm(imgs):   

       
        
        if ".ini" in img:
            continue
   
        x,y = calcPixelPosition(img,xshift,yshift)
   
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


        if output_sigmoid<0.5:
            safePath = os.path.join(aPath,img)
            tile = tile.new_from_image(250).bandjoin(tile[1:3])
            tileMap[y][x]= 1
        elif output_sigmoid>0.5:
            safePath = os.path.join(gPath,img)
           
            tileMap[y][x]= 2

        else:
            safePath = os.path.join(diffPath,img)
            

        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)


        absX,absY = extractTileCoordinates(img)

        absX = absX-xshift
        absY = absY -yshift

        result = result.insert(tile,absX,absY)


       
        
      
        cv2.imwrite(safePath,orig_image)    
        
        
               
    return tileMap, result

def drawResultImage(resultsArray, slideWidth, slideHeight):

     slideWidth = int(slideWidth/500)
     slideHeight = int(slideHeight/500)
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



def makeClassificationRun(tilePath, outPath, model, transform):
    wsis = sortTilesByWSI(tilePath)

    aPath, gPath , diffPath, mapPath,wsiPath  = makeResultFolder(outPath)

    printWsis(wsis)
    
    for slide in tqdm(wsis):
        
        print("processing slide"+ slide)

        slideWidth , slideHeight, xshift, yshift = findWidhHeight(wsis[slide])

        #print("dims of slide " + slide + " with dimensions w: " + str(slideWidth) +" and "+ str(slideHeight))

        w = int(slideWidth/500)
        h = int(slideHeight/500)

        print("dims of out image w: " + str(w)+ "h: " + str(h))

        if slideWidth == 0 or slideHeight == 0:
           print("Warning: the slide "+ slide +" has dims 0 , 0")
           continue
       
       

        tileMap, result = makeTileMap(tilePath,wsis[slide],outPath,slideWidth, slideHeight,xshift,yshift, model, transform)

        

        tifPath = os.path.join(wsiPath,slide+".tif")
    
       
       
        resultImg = drawResultImage(tileMap,slideWidth, slideHeight)

        safePath = os.path.join(mapPath,slide+".jpg")

        resultImg = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)  

        cv2.imwrite(safePath, resultImg)

        result.tiffsave(tifPath, compression='jpeg', 
                  tile=True, tile_width=512, tile_height=512, 
                  pyramid=True,  bigtiff=True)

        


       

if __name__ == '__main__':


     tilePath = r"C:\Users\felix\Desktop\fixedKryoTest\stitcherTest"

     slidePath =r"C:\Users\felix\Desktop\fixedKryoTest\kryoTest"

     outPath = r"C:\Users\felix\Desktop\fixedKryoTest\stitcherRes"


     

     device = ('cuda' if torch.cuda.is_available() else 'cpu')

     transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])

     model = CustomCNN(num_classes=1)
     checkpoint = torch.load(r'C:\Users\felix\Desktop\AutoEncoder\models\model150.pth', map_location=device)
     print('Loading trained model weights...')
     model.load_state_dict(checkpoint['model_state_dict'])
     
     model.eval()
     model = model.to(device)

     makeClassificationRun(tilePath, outPath, model, transform)