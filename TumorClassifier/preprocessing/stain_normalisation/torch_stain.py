import torch
from torchvision import transforms
import torchstain
import cv2
import os

import matplotlib.image as mpimg


def createOutPath(path):
    runPath = os.path.join(path,"result")
    if not os.path.isdir(runPath):
        os.mkdir(runPath)
    mPath = os.path.join(path,"result","macenko")
    if not os.path.isdir(mPath):
        os.mkdir(mPath)
    vPath = os.path.join(path,"result","vahadane")
    if not os.path.isdir(vPath):
        os.mkdir(vPath)

    toCreate = os.path.join(mPath,"normalize")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)

    toCreate = os.path.join(mPath,"hematoxylin")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)

    toCreate = os.path.join(mPath,"eosin")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)

    toCreate = os.path.join(vPath,"normalize")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)

    toCreate = os.path.join(vPath,"hematoxylin")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)

    toCreate = os.path.join(vPath,"eosin")
    if not os.path.isdir(toCreate):
        os.mkdir(toCreate)



def stainNormAllImages(inPath, outPath):
     

     target = cv2.cvtColor(cv2.imread(r"C:\Users\felix\Desktop\kryo\test\Astro\A-N22-2714-K_6000_15000.jpg"), cv2.COLOR_BGR2RGB)
     T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
        ])

     torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
     torch_normalizer.fit(T(target))


     
     for file in os.listdir(inPath):
         d = os.path.join(inPath, file)
         print(file)
         copiedFiles = os.listdir(os.path.join(r"C:\Users\felix\Desktop\Neuro\torchstainKryo",file,"macenko","normalize"))
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

                 normOutPath = os.path.join(r"C:\Users\felix\Desktop\Neuro\torchstainKryo",file,"macenko","normalize",testImg)
                 hOutPath = os.path.join(r"C:\Users\felix\Desktop\Neuro\torchstainKryo",file,"macenko","hematoxylin",testImg)
                 eOutPath = os.path.join(r"C:\Users\felix\Desktop\Neuro\torchstainKryo",file,"macenko","eosin",testImg)
                 
                 print(norm.size())

                 normImg = norm.numpy()
                 hImg = H.numpy()
                 eImg = E.numpy()




                 cv2.imwrite(normOutPath, normImg)
                 cv2.imwrite(hOutPath, hImg)
                 cv2.imwrite(eOutPath, eImg)


def stainNormTest(inPath, outPath, targetPath):
    createOutPath(outPath)
    target = cv2.cvtColor(cv2.imread(targetPath), cv2.COLOR_BGR2RGB)
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
        ])

    torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    torch_normalizer.fit(T(target))


     
    
    copiedFiles = os.listdir(os.path.join(outPath,"result","macenko","normalize"))
       
            
    testImgs = os.listdir(inPath)
            
    for testImg in testImgs:
        if not testImg.endswith(".jpg"):
            continue
        if testImg in copiedFiles:
            print("file " + testImg+ " allready copied")
            continue
        imgPath = os.path.join(inPath,testImg)
        image = cv2.imread(imgPath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t_to_transform = T(image)
        norm, H, E = torch_normalizer.normalize(I=t_to_transform, stains=True)

        normOutPath = os.path.join(outPath,"result","macenko","normalize",testImg)
        hOutPath = os.path.join(outPath,"result","macenko","hematoxylin",testImg)
        eOutPath = os.path.join(outPath,"result","macenko","eosin",testImg)
                 
        print(norm.size())

        normImg = norm.numpy()
        hImg = H.numpy()
        eImg = E.numpy()




        cv2.imwrite(normOutPath, normImg)
        cv2.imwrite(hOutPath, hImg)
        cv2.imwrite(eOutPath, eImg)




if __name__ == '__main__':
    inPath = r"C:\Users\felix\Desktop\neuro\stainTestIn"
    outPath = r"C:\Users\felix\Desktop\neuro\stainTest\torchStain\run2"
    targetPath = r"C:\Users\felix\Desktop\neuro\stainTest\torchStain\run2\O2-N17-1215-K-Q2_119000_16500.jpg"
    stainNormTest(inPath, outPath,targetPath)
    
                