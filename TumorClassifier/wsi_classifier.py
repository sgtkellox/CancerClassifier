import os
import shutil
import random
import math
from sysconfig import get_path

#from preprocessing_slides import process_tiles
#from preprocessing_slides import mask4
#from preprocessing_slides import SIZE

import matplotlib.pyplot as plt

import torch
import cv2
import torchvision.transforms as transforms
import numpy as np

from model import CNNModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

wsiPath = ""
tilesFolder = ""



def makeSplit(wsiPath):
    process_tiles(wsiPath,mask4, tilesFolder)
    return

def calcPixelPosition(image):
    splitP1 = image.split("_")
    x = int(splitP1[1])/500
    y = int(splitP1[2].split(".")[0])/500

    return x , y 


def calcSlideResultWithPositions(tilesFolder):

    

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # list containing all the class labels
    labels = [
        'Astro', 'GBM', 'Oligo'
        ]

    # initialize the model and load the trained weights
    model = CNNModel().to(device)
    checkpoint = torch.load(r'C:\Users\felix\Desktop\Neuro\testModelSmear\model100.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # define preprocess transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    
   
    testImgs = os.listdir(tilesFolder)

    result = []

    for testImg in testImgs:
        if not testImg.endswith(".jpg"):
            continue
        imgPath = os.path.join(tilesFolder,testImg)
        image = cv2.imread(imgPath)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image.to(device))
                    
        output_label = torch.topk(outputs, 1)
              
        pred_class = labels[int(output_label.indices)]
        
        x,y = calcPixelPosition(testImg)

        resEntry = [int(output_label.indices),y,x]

        result.append(resEntry)

    return result
        


                  
def drawResultImage(resultsArray):
     result = np.zeros((540, 1710, 3), np.uint8)
     result.fill(255)

     for entry in resultsArray:
            if entry[0]==0:
                result[int(entry[1])*10:int(entry[1])*10+10,int(entry[2])*10:int(entry[2])*10+10] = [255, 0, 0]
            elif entry[0]==1:
                result[int(entry[1])*10:int(entry[1])*10+10,int(entry[2])*10:int(entry[2])*10+10] = [0, 255, 0]
            elif entry[0]==2:
                result[int(entry[1])*10:int(entry[1])*10+10,int(entry[2])*10:int(entry[2])*10+10] = [0, 0, 255]
     return result
        
       
            
def   blal():        

    if result[0] > result[1]  and result[0] > result[2]:
        precentage = result[0]/(result[0]+result[1]+result[2])
        print("Classified as Astro with certenty of " + str(precentage))
    elif result[1] > result[0]  and result[1] > result[2]:
        precentage = result[1]/(result[0]+result[1]+result[2])
        print("Classified as GBM with certenty of " + str(precentage))
    elif result[2] > result[0]  and result[2] > result[1]:
        precentage = result[2]/(result[0]+result[1]+result[2])
        print("Classified as oligo with certenty of " + str(precentage))

    




def classifySplit(tilesFolder):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # list containing all the class labels
    labels = [
        'Astro', 'GBM', 'Oligo'
        ]

    # initialize the model and load the trained weights
    model = CNNModel().to(device)
    checkpoint = torch.load(r'C:\Users\felix\Desktop\Neuro\testModelSmear\model100.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # define preprocess transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    
   
    testImgs = os.listdir(tilesFolder)
    
    for testImg in testImgs:
        if not testImg.endswith(".jpg"):
            continue
        imgPath = os.path.join(d,testImg)
        image = cv2.imread(imgPath)
        orig_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image.to(device))
                    
        output_label = torch.topk(outputs, 1)
              
        pred_class = labels[int(output_label.indices)]
        resOfCurrentClass.append(pred_class)
        cv2.putText(orig_image, 
        f"GT: {gt_class}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.putText(orig_image, 
            f"Pred: {pred_class}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, (0, 0, 255), 2, cv2.LINE_AA
        )
                
        #if int(output_label.indices) == count:
            #outpath = os.path.join(r"E:\models\results",testImg)
                    
        #else:
            #outpath = os.path.join(r"E:\models\fails",testImg)
        #cv2.imwrite(outpath, orig_image)
                    
    return 

if __name__ == '__main__':
    res = calcSlideResultWithPositions(r"C:\Users\felix\Desktop\Neuro\smearSplitHistNorm\val\Astro")
    print(res)
    img = drawResultImage(res)
    cv2.imwrite(r"F:\N20-1488\filename.png", img)




