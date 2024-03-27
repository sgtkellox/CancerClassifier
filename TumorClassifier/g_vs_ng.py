import os

import matplotlib.pyplot as plt

from tile_creation.filter_utils import open_slide

import torch
import cv2
import torchvision.transforms as transforms
import numpy as np

from modell_training.effNet_v2.effNet_model import build_model
from sklearn  import metrics
from sklearn.metrics import ConfusionMatrixDisplay

from tile_creation.tile_utils import calcPixelPosition


from modell_training.simpleNet.model import CNNModel

from tqdm import tqdm

glial_labels = ["A","EPN", "GBM", "H3" ,"O" ,"PA","PXA"]
ng_labels = ["LYM", "MB", "MEL", "MEN" ,"MET" , "PIT", "SCHW"]


labels = ["glial","non-glial"]


def sortTilesByWSI(path):

    wsis = {}

    for img in os.listdir(path):

        wsiName = img.split("_")[0]

        if wsiName in wsis:
            wsis[wsiName].append(img)
        else:
            wsis[wsiName] = []
            wsis[wsiName].append(img)
    #print("finished sorting by wsi")
    return wsis


def getDiagFromSlide(slide):
    if slide.split("-")[0].startswith("A"):
        gt_class_name = "glial"
    elif slide.split("-")[0].startswith("G"):
        gt_class_name = "glial"
    elif slide.split("-")[0].startswith("O"):
        gt_class_name = "glial"
    elif slide.split("-")[0].startswith("EPN"):
        gt_class_name = "glial"
    elif slide.split("-")[0] in ng_labels:
        gt_class_name = "non-glial"
    elif slide.split("-")[0] in glial_labels:
        gt_class_name = "glial"
    return gt_class_name


def getIndexFromLabel(label):
    return labels.index(label)

def extractTileCoordinates(image):

    splitP1 = image.split("_")
    x = int(splitP1[1])
    y = int(splitP1[2].split(".")[0])
    return x , y

def extractXCoordinate(tile):
    splitP1 = tile.split("_")
    x = int(splitP1[1])
    return x

def findWidhHeight(images,tileSize):
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

    
    width = maxX + 1000-minX
    height = maxY + 1000 - minY
    
    return width, height , xshift, yShift

def makeTileMap(tilePath, imgs, imagesOutPath ,slideWidth, slideHeight,tileSize, xshift, yshift, model, transform,showImages=False):

    tileMap = np.zeros((slideHeight+1, slideWidth+1, 1), np.uint8)

    if imgs[0].split("-")[0].startswith("A"):
        gt_class_name = "glial"
    elif imgs[0].split("-")[0].startswith("G"):
        gt_class_name = "glial"
    elif imgs[0].split("-")[0].startswith("O"):
        gt_class_name = "glial"
    elif imgs[0].split("-")[0].startswith("EPN"):
        gt_class_name = "glial"
    elif imgs[0].split("-")[0] in ng_labels:
        gt_class_name = "non-glial"
    elif imgs[0].split("-")[0] in glial_labels:
        gt_class_name = "glial"
        

   
    right = 0;
  
    for img in tqdm(imgs):   
        if ".ini" in img:
            continue


        #print("Classifying " + str(img))
        
        x,y = calcPixelPosition(img,xshift,yshift,tileSize)

        x = int(x)
        y = int(y)
    
        imgFullPath = os.path.join(tilePath,img)
        image = cv2.imread(imgFullPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if showImages:
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
        #print(img+ " pred  "+ pred_class_name+ " gt "+ gt_class_name)

        if showImages:

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
            #print("safepath "+safepath)
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(safepath, orig_image)
        
    acc = right/len(imgs)
    
    

        
    return tileMap, acc, 

     
def getWsiDimensions(nNumber, slidePath, lvl=0):
    slides = os.listdir(slidePath)
   
    for wsi in slides:
        wsiSplit = wsi.split(".")[0].split("-")

        wsiNumber = wsiSplit[0] + "-" + wsiSplit[1] + "-" + wsiSplit[2] + "-" + wsiSplit[3] + "-" + wsiSplit[4]

        #print("slide from slide folder "+ wsiNumber)

      
        
        if wsiNumber == nNumber:
               
            slidePath = os.path.join(slidePath,wsi)
            
            slide = open_slide(slidePath)
            w = int(slide.properties["openslide.level[1].width"])
            h = int(slide.properties["openslide.level[1].height"])
            return w , h
                    
    return 0, 0




def drawGLialResultImage(resultsArray, slideWidth, slideHeight):
    result = np.zeros((slideHeight*20+200, slideWidth*20+350, 3), np.uint8)
    result.fill(255)

    constX = 100
    constY = 50    
    
    nums = [0] * 3
     
    diagColorMap = {
        }
    diagColorMap["glial"] = [255,0,0]
    diagColorMap["non-glial"] = [0, 255, 0]
    
     
     

    for x in range(len(resultsArray)):
        for y in range(len(resultsArray[0])):
             
            if resultsArray[x][y]==1:
                nums[1] +=1
                result[x*20+constX:x*20+20+constX,y*20+constY:y*20+20+constY] = diagColorMap["glial"]
            elif resultsArray[x][y]==2:
                nums[2] +=1
                result[x*20+constX:x*20+20+constX,y*20+constY:y*20+20+constY] = diagColorMap["non-glial"]
            
               

    
           
    return result
    
    


               



def makeClassificationRun(tilePath, outPath, imagesOutPath, model, transform, tileSize):
    gts = []
    results = []
    total = 0
    for root, dirs, files in os.walk(tilePath):
        total += len(files)
    processed = 0
        
    #pbar = tqdm(desc = "slides processed", total = total)
    for folder in os.listdir(tilePath):       
        folderPath = os.path.join(tilePath,folder)
        if os.path.isfile(folderPath):
            continue
        wsis = sortTilesByWSI(folderPath)
        for slide in wsis:
            
            diagName = getDiagFromSlide(slide)
            #diagNR = getIndexFromLabel(diagName)
            gts.append(diagName)
            width, height , xshift, yShift = findWidhHeight(wsis[slide],tileSize)
            
        
            print("dims of slide " + slide + " with dimensions w: " + str(width) +" and "+ str(height))
            if width == 0 or height == 0:
               print("Warning: the slide "+ slide +" has dims 0 , 0")
               continue
            slideWidth = int(width/tileSize) 
            slideHeight = int(height/tileSize) 

            #print("width "+ str(width))
            #print("height "+ str(height))
            #print("xshift "+ str(xshift))
            #print("yshift "+ str(yShift))
            
            tileMap, acc = makeTileMap(folderPath,wsis[slide],imagesOutPath,slideWidth, slideHeight,tileSize, xshift, yShift, model, transform,showImages=False)
            flatMap = tileMap.flatten()
            
            res = np.bincount(flatMap)
            
            res = res[1:]
            
             
            classNr = np.where(res == max(res))[0][0]
            print("classNr " + str(classNr))
            
            results.append(labels[classNr])
            
            
            
            #print(res)
            resultImg = drawGLialResultImage(tileMap,slideWidth, slideHeight)

            safePath = os.path.join(outPath,slide+".jpg")

            #print(slide + str(acc))

            resultImg = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)

            cv2.imwrite(safePath, resultImg)
            processed+=1
            #pbar.update(processed)

            print(results)
            print(gts)
            
            

    confusion_matrix = metrics.confusion_matrix(gts, results, labels=labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = labels) 
    cm_display.plot()
    confMatrixPath = os.path.join(outPath,"confMatrix.jpg")
    plt.savefig(confMatrixPath)
    plt.close()
       

if __name__ == '__main__':


     tilePath = r"E:\testSets\smear\other\test"

     

     outPath = r"E:\result\smear\other\384_10x"


     
     
     mapsPath = os.path.join(outPath,"maps") 
     imagesPath = os.path.join(outPath,"images")
     os.mkdir(mapsPath) 
     os.mkdir(imagesPath)


     tileSize = 384
     

     device = ('cuda' if torch.cuda.is_available() else 'cpu')

     transform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
            )
    ])

     model = build_model(pretrained=True, fine_tune=True, num_classes=2)
     
     
     checkpoint = torch.load(r'E:\models\smear\other\load\model_64.pth', map_location=device)

     model.load_state_dict(checkpoint['model_state_dict'])
     
     model.eval()
     model = model.to(device)

     makeClassificationRun(tilePath, mapsPath,imagesPath, model, transform, tileSize)




    




