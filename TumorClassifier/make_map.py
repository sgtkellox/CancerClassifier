
import torch
import cv2
import glob as glob
import os
from modell_training.binary_classifier.model import CustomCNN
from torchvision import transforms

import numpy as np

from tile_creation.tile_utils import  calcPixelPosition
#from tile_creation.filter_utils import open_slide

import shutil

labels = ["artifact","good"]

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
    print(image)
    x = int(splitP1[1])
    y = int(splitP1[2].split(".")[0])
    return x , y



def makeTileMap(tilePath, imgs, imagesOutPath ,slideWidth, slideHeight, model, transform, xShift,yShift):

    tileMap = np.zeros((slideHeight, slideWidth, 1), np.uint8)
    
  
    for img in imgs:   
        if ".ini" in img:
            continue

        #print("Classifying " + str(img))
       
        x,y = calcPixelPosition(img,xShift,yShift,tileSize)

        x = int(x)
        y = int(y)
        
        imgFullPath = os.path.join(tilePath,img)
        image = cv2.imread(imgFullPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = transform(image)
       
        image = torch.unsqueeze(image, 0)
        image = image.to(device)
    
        # Forward pass throught the image.
        outputs = model(image)
      
        output_sigmoid = torch.sigmoid(outputs)
        pred_class = 1 if output_sigmoid > 0.5 else 0

        if pred_class == 0:
            artefactFolder = os.path.join(imagesOutPath,"artefact")
            if not os.path.isdir(artefactFolder):
                os.mkdir(artefactFolder)
            safePath = os.path.join(artefactFolder,img)     
        else:
            goodFolder = os.path.join(imagesOutPath,"good")
            if not os.path.isdir(goodFolder):
                os.mkdir(goodFolder)
            safePath = os.path.join(goodFolder,img)
        shutil.move(imgFullPath,safePath)
        
        tileMap[y][x] = pred_class +1
             
    return tileMap


def getWsiDimensions(nNumber, slidePath):
    slides = os.listdir(slidePath)
   
    for wsi in slides:
        wsiSplit = wsi.split(".")[0].split("-")

        wsiNumber = wsiSplit[0] + "-" + wsiSplit[1] + "-" + wsiSplit[2] + "-" + wsiSplit[3] + "-" + wsiSplit[4]
      
        if wsiNumber == nNumber:
               
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

    
    width = maxX+2*tileSize-minX
    height = maxY + 2*tileSize - minY


    return width, height , xshift, yShift


                  
def drawResultImage(resultsArray, slideWidth, slideHeight):
     result = np.zeros((slideHeight*10, slideWidth*10, 3), np.uint8)
     result.fill(255)

     for x in range(len(resultsArray)):
         for y in range(len(resultsArray[0])):
             
             if resultsArray[x][y]==1:
                result[x*10:x*10+10,y*10:y*10+10] = [255, 0, 0]
             elif resultsArray[x][y]==2:
                result[x*10:x*10+10,y*10:y*10+10] = [0, 255, 0]
            
     return result

def collectAllreadyProcessed(path):
    wsis = []

    for img in os.listdir(path):

        split = img.split(".")[0]
        split = split.split("-")
        wsiName = split[0] + "-"+split[1] + "-" + split[2]+ "-"+split[3]+"-"+split[4]
        
        print(wsiName)
        wsis.append(wsiName)
                     
    return wsis



def makeClassificationRun(tilePath, slidePath, outPath, imagesOutPath, model, transform):
    wsis = sortTilesByWSI(tilePath)
    allreadyProcessed = collectAllreadyProcessed(outPath)
    for slide in wsis:
        if slide in allreadyProcessed:
            print("allready done: " + slide )
            continue
        print("slide from tileName: "+slide)
        slideWidth , slideHeight, xShift, yShift = findWidhHeight(wsis[slide])
        print("dims of slide " + slide + " with dimensions w: " + str(slideWidth) +" and "+ str(slideHeight))
        if slideWidth == 0 or slideHeight == 0:
           print("Warning: the slide "+ slide +" has dims 0 , 0")
           continue
        slideWidth = int(slideWidth/tileSize) 
        slideHeight = int(slideHeight/tileSize) 
        
        #slideWidth = int(slideWidth - (slideWidth % 500))
        #slideHeight = int(slideHeight - (slideHeight % 500))

        tileMap = makeTileMap(tilePath,wsis[slide],imagesOutPath,slideWidth, slideHeight, model, transform,xShift, yShift)
       
        resultImg = drawResultImage(tileMap,slideWidth, slideHeight)

        safePath = os.path.join(outPath,slide+".jpg")
       
        resultImg = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)

        cv2.imwrite(safePath, resultImg)





if __name__ == '__main__':

    tilePath = r"F:\home\rerun\images\good"

    slidePath =r"F:\slides\kryoQ2"

    outPath = r"C:\Users\felix\Desktop\res\maps"


    imagesOutPath = r"C:\Users\felix\Desktop\res\imgs"

    tileSize = 512

    IMAGE_SIZE = 224
    device = 'cuda'
    # Load the trained model.
    model = CustomCNN(num_classes=1)
    checkpoint = torch.load(r'C:\Users\felix\Desktop\AutoEncoder\models4\25.pth', map_location=device)
    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ]) 
    model.eval()

    model = model.to(device)


    makeClassificationRun(tilePath, slidePath, outPath,imagesOutPath, model, transform)
        
