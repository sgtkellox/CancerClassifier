import os

import matplotlib.pyplot as plt

from tile_creation.filter_utils import open_slide

import torch
import cv2
import torchvision.transforms as transforms
import numpy as np

from modell_training.effNet.effNet_model import build_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from tile_creation.tile_utils import calcPixelPosition

from modell_training.simpleNet.model import CNNModel

labels = ['Astro', 'GBM', 'Oligo']



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

def makeTileMap(tilePath, imgs, imagesOutPath ,slideWidth, slideHeight, model, transform):

    tileMap = np.zeros((slideHeight, slideWidth, 1), np.uint8)

    if imgs[0].split("-")[0].startswith("A"):
        gt_class_name = "Astro"
    elif imgs[0].split("-")[0].startswith("G"):
        gt_class_name = "GBM"
    elif imgs[0].split("-")[0].startswith("O"):
        gt_class_name = "Oligo"

   
    right = 0;
  
    for img in imgs:   
        if ".ini" in img:
            continue


        print("Classifying " + str(img))

        
        
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
        outputs = outputs.detach().cpu().numpy()
        pred_class =np.argmax(outputs[0])
        pred_class_name = labels[pred_class]
        tileMap[y][x] = pred_class +1
        if pred_class_name == gt_class_name:
            right +=1
        print(img+ " pred  "+ pred_class_name+ " gt "+ gt_class_name)

        cv2.putText(
            orig_image, f"GT: {gt_class_name}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA
        )
    # Annotate the image with prediction.
        cv2.putText(
            orig_image, f"Pred: {pred_class_name.lower()}",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (100, 100, 225), 2, lineType=cv2.LINE_AA
        ) 
        safepath = os.path.join(imagesOutPath,img)
        print("safepath "+safepath)
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(safepath, orig_image)
        
        
    
         

    acc = right/len(imgs)

        
    return tileMap, acc


def getWsiDimensions(nNumber, slidePath):
    slides = os.listdir(slidePath)
   
    for wsi in slides:
        wsiSplit = wsi.split(".")[0].split("-")

        wsiNumber = wsiSplit[0] + "-" + wsiSplit[1] + "-" + wsiSplit[2] + "-" + wsiSplit[3] + "-" + wsiSplit[4]

        print("slide from slide folder "+ wsiNumber)

      
        
        if wsiNumber == nNumber:
               
            slidePath = os.path.join(slidePath,wsi)
            
            slide = open_slide(slidePath)
            a = slide.dimensions
            return a[0] , a[1]
                    
    return 0, 0


                  
def drawResultImage(resultsArray, slideWidth, slideHeight):
     result = np.zeros((slideHeight*10, slideWidth*10, 3), np.uint8)
     result.fill(255)

     numAstro=0
     numGBM = 0
     numOligo= 0

     for x in range(len(resultsArray)):
         for y in range(len(resultsArray[0])):
             
             if resultsArray[x][y]==1:
                numAstro +=1
                result[x*10:x*10+10,y*10:y*10+10] = [255, 0, 0]
             elif resultsArray[x][y]==2:
                numGBM +=1
                result[x*10:x*10+10,y*10:y*10+10] = [0, 255, 0]
             elif resultsArray[x][y]==3:
                numOligo +=1
                result[x*10:x*10+10,y*10:y*10+10] = [0, 0, 255]

     cv2.putText(
        result, "Astro " + str(numAstro),
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255, 0, 0), 2, lineType=cv2.LINE_AA
     )
    # Annotate the image with prediction.
     cv2.putText(
        result, "GBM " + str(numGBM),
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA
     )
     cv2.putText(
        result, "Oligo " + str(numOligo),
        (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 0, 255), 2, lineType=cv2.LINE_AA
      ) 
    
     
     return result


def makeClassificationRun(tilePath, slidePath, outPath, imagesOutPath, model, transform):
    wsis = sortTilesByWSI(tilePath)
    for slide in wsis:
        print("slide from tileName: "+slide)
        slideWidth , slideHeight = getWsiDimensions(slide,slidePath)
        print("dims of slide " + slide + " with dimensions w: " + str(slideWidth) +" and "+ str(slideHeight))
        if slideWidth == 0 or slideHeight == 0:
           print("Warning: the slide "+ slide +" has dims 0 , 0")
           continue
        slideWidth = int(slideWidth/500) 
        slideHeight = int(slideHeight/500) 
        #slideWidth = int(slideWidth - (slideWidth % 500))
        #slideHeight = int(slideHeight - (slideHeight % 500))

        tileMap, acc = makeTileMap(tilePath,wsis[slide],imagesOutPath,slideWidth, slideHeight, model, transform)
       
        resultImg = drawResultImage(tileMap,slideWidth, slideHeight)

        safePath = os.path.join(outPath,slide+".jpg")

        print(slide + str(acc))

        resultImg = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)

        cv2.imwrite(safePath, resultImg)


       

if __name__ == '__main__':


     tilePath = r"C:\Users\felix\Desktop\neuro\kryoTestTiles"

     slidePath =r"C:\Users\felix\Desktop\neuro\kryoTest"

     outPath = r"C:\Users\felix\Desktop\neuro\testing\maps"


     imagesOutPath = r"C:\Users\felix\Desktop\neuro\testing\results"

     

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

     model = build_model(pretrained=True, fine_tune=True, num_classes=3)
     
     
     checkpoint = torch.load(r'C:\Users\felix\Desktop\neuro\models\model_14_pretrained.pth', map_location=device)

     model.load_state_dict(checkpoint['model_state_dict'])
     
     model.eval()
     model = model.to(device)

     makeClassificationRun(tilePath, slidePath, outPath,imagesOutPath, model, transform)




    




