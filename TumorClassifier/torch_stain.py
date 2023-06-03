import torch
from torchvision import transforms
import torchstain
import cv2
import os

import matplotlib.image as mpimg



if __name__ == '__main__':
     path = r"D:\kyroSplit"

     target = cv2.cvtColor(cv2.imread(r"D:\kyroSplit\GBM\GBM-N17-0592-K_62500_34500.jpg"), cv2.COLOR_BGR2RGB)
     T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
        ])

     torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
     torch_normalizer.fit(T(target))


     
     for file in os.listdir(path):
         d = os.path.join(path, file)
         print(file)
         copiedFiles = os.listdir(os.path.join(r"C:\Users\felix\Desktop\neuro\augmentOutput",file,"torchstain","macenko"))
         if os.path.isdir(d):
            
             testImgs = os.listdir(d)
            
             for testImg in testImgs:
                 if not testImg.endswith(".jpg"):
                     continue
                 if testImg in copiedFiles:
                    print("file " + testImg+ " allready copied")
                    continue
                 imgPath = os.path.join(d,testImg)
                 image = cv2.imread(imgPath)
                 #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                 t_to_transform = T(image)
                 norm, H, E = torch_normalizer.normalize(I=t_to_transform, stains=True)

                 normOutPath = os.path.join(r"C:\Users\felix\Desktop\neuro\augmentOutput",file,"torchstain","macenko",testImg)
                 hOutPath = os.path.join(r"C:\Users\felix\Desktop\neuro\augmentOutput",file,"torchstain","h",testImg)
                 eOutPath = os.path.join(r"C:\Users\felix\Desktop\neuro\augmentOutput",file,"torchstain","e",testImg)
                 
                 print(norm.size())

                 normImg = norm.numpy()
                 hImg = H.numpy()
                 eImg = E.numpy()




                 cv2.imwrite(normOutPath, normImg)
                 cv2.imwrite(hOutPath, hImg)
                 cv2.imwrite(eOutPath, eImg)
                