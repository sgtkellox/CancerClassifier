import os 

import torch
import cv2

from model import CustomCNN
from torchvision import transforms
import argparse

import argparse
import shutil



def makeFilterRun(path,thresh, model):
    
    imgs = os.listdir(path) 
    for img in imgs:   
        if ".ini" in img:
            continue
       
        imgFullPath = os.path.join(path,img)
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
            threshPath = os.path.join(thresh,img)
            
            shutil.move(imgFullPath,threshPath)
        
             

    
    return 


def checkSizeTileWise(path):
    
    return 
    

if __name__ == '__main__':
    
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-p", "--path", help="The path to the folder containing the images")
    argParser.add_argument("-b", "--bin", help="The path to the bin")
    argParser.add_argument("-m", "--model", help="The path to the model")
    
    args = argParser.parse_args()

    path = args.path
    thresh = args.bin
    modelPath = args.model
    

    IMAGE_SIZE = 224
    device = 'cuda'
    # Load the trained model.
    model = CustomCNN(num_classes=1)
    checkpoint = torch.load(modelPath, map_location=device)
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
    
    makeFilterRun(path,thresh, model)