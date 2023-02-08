import slideio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math 


pathToOutputFolder = ""

def isBackGround(image):
    mean = np.mean(image, axis=(0, 1, 2))

    if mean > 100 and mean < 200:
        return False
    else:
       return True

#Brightness need be selected from [0,100], contrast from [1,3]       
def ChangeBrightnessAndContrast(imgage, brightness , contrast ):
    alteredImage = np.zeros(image.shape, image.dtype)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                alteredImage[y,x,c] = np.clip(contrast*image[y,x,c] + brightness, 0, 255)
    return alteredImage
 

def minMaxStretch(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    min = hsvImage[..., 2].min()
    max = hsvImage[..., 2].max()
    minmax_img = np.zeros([hsvImage.shape[0], hsvImage.shape[1], 3],dtype = 'uint8')
    for i in range(hsvImage.shape[0]):
        for j in range(hsvImage.shape[1]):
            minmax_img[i,j,2] = 255*(hsvImage[i,j,2]-min)/(max-min)

    rgbimg = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2RGB)
    return rgbimg

def lutContrastEnhancement(image):
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    image = cv2.LUT(image, table)
    return image

def histogramEqualisationOnVChannel(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsvImage[:, :, 2] = cv2.equalizeHist(hsvImage[:, :, 2])
    equalizedImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2RGB)
    return equalizedImage

def histogramEqalisation(image):
    bgrImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ycrcbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2YCrCb)
    ycrcbImage[:, :, 0] = cv2.equalizeHist(ycrcbImage[:, :, 0])
    equalizedImage = cv2.cvtColor(ycrcbImage, cv2.COLOR_YCrCb2BGR)
    return equalizedImage

def makeSlices(sizeOfSlice, wsiWidth, wsiHeight, scene):
    scliceCols = math.floor(wsiHeight/sizeOfSlice)
    scliceRows = math.floor(wsiWidth/sizeOfSlice)
    i = 0
    j = 0
    count = 1;
    for i in range(scliceRows):
        for j in range(scliceCols):
            minX = i*sizeOfSlice
            
            minY = j*sizeOfSlice
           
            image = scene.read_block((minX,minY,sizeOfSlice,sizeOfSlice), (sizeOfSlice,sizeOfSlice))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if not isBackGround(image):

                contrastEnhancedImage = histogramEqalisation(image)

                outpath = r"C:\Users\felix\Desktop\croppedImages2\crop" + str(count) + ".jpg"
                cv2.imwrite(outpath, image)

                #outpathEnhandced = r"C:\Users\felix\Desktop\ContrastEnhancment\enhancedCrop" + str(count) + ".jpg"
                #cv2.imwrite(outpathEnhandced, contrastEnhancedImage)

                count +=1
    
    return





path = r"D:\Ramin_SS_Oligo_Astro\A2-N17-1152Q.svs"

slide = slideio.open_slide(path,'SVS')
num_scenes = slide.num_scenes
scene = slide.get_scene(0)

makeSlices(1000, scene.rect[2], scene.rect[3], scene)






