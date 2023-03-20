import shutil
import torch
import cv2
import torchvision.transforms as transforms
import argparse
import os
from model import CNNModel
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def makeTestRun(path):
    

    # the computation device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # list containing all the class labels
    labels = [
        'Astro', 'GBM', 'Oligo'
        ]

    # initialize the model and load the trained weights
    model = CNNModel().to(device)
    checkpoint = torch.load(r'D:\ClassifierResults\resNet50Kryo\models\model115.pth', map_location=device)
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
    count = 0
    res = []
    gt = []
    
    for file in os.listdir(path):
        d = os.path.join(path, file)
        if os.path.isdir(d):
            resOfCurrentClass = []
            gt_class = file
            testImgs = os.listdir(d)
            gtOfCurrentClass = [file] * len(testImgs)
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
                
                if int(output_label.indices) == count:
                    outpath = os.path.join(r"D:\ClassifierResults\resNet50Kryo\succes",testImg)
                    
                else:
                    outpath = os.path.join(r"D:\ClassifierResults\resNet50Kryo\fail",testImg)
                cv2.imwrite(outpath, orig_image)
                    
            count+=1
            res+=resOfCurrentClass
            gt+= gtOfCurrentClass

    return res, gt


def calcMetric(res, gt):
    wrong = 0
    right = 0
    for i in range(len(res)):
        if res[i] != gt[i]:
            wrong+=1
        else:
            right+=1
    
    posRatio = right/len(res)
    return posRatio

res , gt = makeTestRun(r"C:\Users\felix\Desktop\Neuro\KryoSplit\test")
acc = calcMetric(res,gt)
cm = confusion_matrix(gt, res, labels=['Astro','GBM','Oligo'])
cmd = ConfusionMatrixDisplay(cm, display_labels=['Astro','GBM','Oligo'])
cmd.plot()
plt.savefig(r'D:\ClassifierResults\resNet50Kryo\cf.jpg')
plt.show()
