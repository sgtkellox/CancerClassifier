import numpy as np
import cv2
import torch
import glob as glob
from torchvision import transforms
from torch.nn import functional as F
from torch import topk
from model import CNNModel
import tifffile
from wsi_utils import calcPixelPosition
import matplotlib.pyplot as plt

import os

labels = ['Astro','GBM','Oligo']



# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def show_cam(CAMs, width, height, orig_image, class_idx, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + orig_image * 0.5
        # put class label text on the result
        cv2.putText(result, str(int(class_idx[i])), (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(save_name, result)

 
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


def makeGradCamForFolder(path):

    labels = [
        'Astro', 'GBM', 'Oligo'
        ]


    # define computation device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize model, switch to eval model, load trained weights
    model = CNNModel()
    checkpoint = torch.load(r'C:\Users\felix\Desktop\Neuro\testModelSmear\model100.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    for file in os.listdir(path):
        d = os.path.join(path, file)
        if os.path.isdir(d):
            
            testImgs = os.listdir(d)
            
            for testImg in testImgs:
                if not testImg.endswith(".jpg"):
                    continue
                # read the image
                imgPath = os.path.join(d,testImg)
                image = cv2.imread(imgPath)
                orig_image = image.copy()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #image = np.expand_dims(image, axis=2)
                height, width, _ = orig_image.shape
                # apply the image transforms

                image_tensor = transform(image)

                features_blobs = []
                def hook_feature(module, input, output):
                    features_blobs.append(output.data.cpu().numpy())
                model._modules.get('conv4').register_forward_hook(hook_feature)

                # get the softmax weight
                params = list(model.parameters())
                weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

                # add batch dimension
                image_tensor = image_tensor.unsqueeze(0)
                image_tensor = image_tensor.to(device)
                # forward pass through model
                outputs = model(image_tensor)
                # get the softmax probabilities
                probs = F.softmax(outputs).data.squeeze()


                # get the class indices of top k probabilities
                class_idx = topk(probs, 1)[1].int()

    
                # generate class activation mapping for the top1 prediction
                CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
                # file name to save the resulting CAM image with
                imgName, file_extension = os.path.splitext(testImg)
                save_name = os.path.join(r"C:\Users\felix\Desktop\Neuro\SmearSplitModelResult",imgName+"cam.jpg")
                # show and save the results
                show_cam(CAMs, width, height, orig_image, class_idx, save_name)

def drawCAMTiff(camFolder,slideWidth, slideHeight):

    camMap = [["empty"for row in range(int(slideHeight/500))] for col in range(int(slideWidth/500))]
    cams = os.listdir(camFolder)

    for cam in cams:
        x,y = calcPixelPosition(cam)
        #print("X "+ str(x)+ " Y "+str(y))
        camMap[x][y] = cam
    filler = np.zeros((slideHeight, 1, 3), np.uint8)
    for i in range(len(camMap)):
        if camMap[i][0] == "empty":
            row = np.zeros((500, 500, 3), np.uint8)
        for j in range(1,len(camMap[0])):
            if not camMap[i][j] == "empty":
                camPath = os.path.join(camFolder,camMap[i][j])
                image = cv2.imread(camPath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                row = np.vstack((row,image)) 
            else:
               placeholder = np.zeros((500, 500, 3), np.uint8)
               row = np.vstack((row,placeholder))
        filler = np.hstack((filler,row))
    
    print(filler.shape)           
    tifffile.imwrite(r'E:\StitchedCams\cams.tiff', filler,  photometric='rgb')
    return 

def showTiff(tifPath):
    I = plt.imread(tifPath)
    plt.imshow(I)
    plt.show()

    return


if __name__ == '__main__':
    makeGradCamForFolder(r"C:\Users\felix\Desktop\Neuro\smearSplitHistNorm\val")