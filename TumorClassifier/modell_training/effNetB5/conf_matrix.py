import torch
import cv2
import numpy as np
import glob as glob
import os

from effNet_model import build_model
from torchvision import transforms

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Constants.

# Constants.
DATA_PATH = r'C:\Users\felix\Desktop\neuro\kryo\test'
IMAGE_SIZE = 224
DEVICE = 'cuda'

# Class names.
class_names = ['Astro','GBM', 'Oligo']


def sortTilesByWSI(images):

    wsis = {}

    for img in images:

        wsiName = img.split("_")[0]

        if wsiName in wsis:
            wsis[wsiName].append(img)
        else:
            wsis[wsiName] = []
            wsis[wsiName].append(img)
    print("finished sorting by wsi")
    return wsis




if __name__ == '__main__':

    model = build_model(pretrained=True, fine_tune=True, num_classes=3)
    checkpoint = torch.load(r"C:\Users\felix\Desktop\neuro\models\model_14_pretrained.pth", map_location=DEVICE)
    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)



    model.eval()


    # Iterate over all the images and do forward pass.

    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ])

    gtArray = []

    predArray = []

    testSize = 0

    right = 0

    for file in os.listdir(DATA_PATH):
            d = os.path.join(DATA_PATH, file)
            if os.path.isdir(d):
                resOfCurrentClass = []
                gt_class = file
                testImgs = os.listdir(d)
                gtOfCurrentClass = [file] * len(testImgs)
                testSize +=len(testImgs)
                wsis = sortTilesByWSI(testImgs)
                for wsi in wsis:

                    wsiGt = []
                    wsiGt = [file]*len(wsis[wsi])

                    wsiPred = []
                    for imgName in wsis[wsi]:

               
                        imgPath = os.path.join(d,imgName)
                        image = cv2.imread(imgPath)
                
                        print("classifying " +imgPath)
                
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                        image = transform(image)
                        image = torch.unsqueeze(image, 0)
                        image = image.to(DEVICE)
    
                        # Forward pass throught the image.
                        outputs = model(image)
                        outputs = outputs.detach().cpu().numpy()
                        pred_class_name = class_names[np.argmax(outputs[0])]
                        resOfCurrentClass.append(pred_class_name)
                        wsiPred.append(pred_class_name)
               
                        if pred_class_name == gtOfCurrentClass:
                            right+=1
    
                    # Annotate the image with ground truth.
                    print("pred")
                    print(wsiPred)
                    print("gt")
                    print(wsiPred)
                    cm = confusion_matrix(wsiGt, wsiPred, labels=class_names)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
                    disp.plot()


                    plt.savefig(r"C:\Users\felix\Desktop\neuro\model_output\cf\cf"+wsi+".jpg")
            
            gtArray = gtArray+gtOfCurrentClass
            predArray = predArray+resOfCurrentClass
            

    cm = confusion_matrix(gtArray, predArray, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot()


    plt.savefig(r"C:\Users\felix\Desktop\neuro\model_output\cf\cf.jpg")
    acc = right/testSize

    print("acc "+ str(acc))
    plt.show()