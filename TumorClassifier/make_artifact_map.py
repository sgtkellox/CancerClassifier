import os
import argparse
import math


import numpy as np

import torch
import cv2
import torchvision.transforms as transforms


from tile_creation.tile_utils import calcPixelPosition

from tile_creation.filter_utils import open_slide

from modell_training.binary_classifier.model import CustomCNN

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

def makeTileMap(tilePath, imgs, imagesOutPath ,slideWidth, slideHeight, model, transform):

    tileMap = np.zeros((slideHeight, slideWidth, 1), np.uint8)

    aPath = os.path.join(imagesOutPath,"artefact")
    if not os.path.isdir(aPath):
        os.mkdir(aPath)

    gPath = os.path.join(imagesOutPath,"good")
    if not os.path.isdir(gPath):
        os.mkdir(gPath)

    diffPath = os.path.join(imagesOutPath,"difficult")
    if not os.path.isdir(diffPath):
        os.mkdir(diffPath)

  
    for img in imgs:   
        if ".ini" in img:
            continue


        #print("Classifying " + str(img))

        
        
        x,y = calcPixelPosition(img)

        x = int(x)
        y = int(y)
        
        imgFullPath = os.path.join(tilePath,img)
        image = cv2.imread(imgFullPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_image = image.copy()
        image = transform(image)
       
        image = torch.unsqueeze(image, 0)
        image = image.to(device)


        # Forward pass throught the image.
        outputs = model(image)
        output_sigmoid = torch.sigmoid(outputs)


        if output_sigmoid<0.02:
            safePath = os.path.join(aPath,img)
            #print(safePath)
            #print(output_sigmoid)
            tileMap[y][x]= 1
        elif output_sigmoid>0.8:
            safePath = os.path.join(gPath,img)
            #print(safePath)
            #print(output_sigmoid)
            tileMap[y][x]= 2

        else:
            safePath = os.path.join(diffPath,img)
            

        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        
      
        cv2.imwrite(safePath,orig_image)
    
        
               
    return tileMap

def drawResultImage(resultsArray, slideWidth, slideHeight):
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
    






def makeClassificationRun(tilePath, slidePath, outPath, imagesOutPath, model, transform):
    wsis = sortTilesByWSI(tilePath)
    for slide in wsis:
        #print("slide from tileName: "+slide)
        slideWidth , slideHeight = getWsiDimensions(slide,slidePath)

        print("dims of slide " + slide + " with dimensions w: " + str(slideWidth) +" and "+ str(slideHeight))
        if slideWidth == 0 or slideHeight == 0:
           print("Warning: the slide "+ slide +" has dims 0 , 0")
           continue
        slideWidth = int(slideWidth/500) 
        slideHeight = int(slideHeight/500) 
        #slideWidth = int(slideWidth - (slideWidth % 500))
        #slideHeight = int(slideHeight - (slideHeight % 500))

        tileMap = makeTileMap(tilePath,wsis[slide],imagesOutPath,slideWidth, slideHeight, model, transform)
       
        resultImg = drawResultImage(tileMap,slideWidth, slideHeight)

        safePath = os.path.join(outPath,slide+".jpg")

        resultImg = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)

        cv2.imwrite(safePath, resultImg)


       

if __name__ == '__main__':


     tilePath = r"C:\Users\felix\Desktop\AutoEncoder\tiles"

     slidePath =r"C:\Users\felix\Desktop\AutoEncoder\slide"

     outPath = r"C:\Users\felix\Desktop\AutoEncoder\map2"


     imagesOutPath = r"C:\Users\felix\Desktop\AutoEncoder\classified2"

     

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
     checkpoint = torch.load(r'C:\Users\felix\Desktop\AutoEncoder\models\model85.pth', map_location=device)
     print('Loading trained model weights...')
     model.load_state_dict(checkpoint['model_state_dict'])
     
     model.eval()
     model = model.to(device)

     makeClassificationRun(tilePath, slidePath, outPath,imagesOutPath, model, transform)