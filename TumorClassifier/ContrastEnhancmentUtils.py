import numpy as np
import cv2
import math 
import matplotlib.pyplot as plt


class ContrastEnhancmentUtils:
    """description of class"""

    def __init__(self):
        name = ""

    #Brightness need be selected from [0,100], contrast from [1,3]       
    def ChangeBrightnessAndContrast(self,imgage, brightness , contrast ):
        alteredImage = np.zeros(image.shape, image.dtype)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    alteredImage[y,x,c] = np.clip(contrast*image[y,x,c] + brightness, 0, 255)
        return alteredImage
 

    def minMaxStretch(self,image):
        hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        min = hsvImage[..., 2].min()
        max = hsvImage[..., 2].max()
        minmax_img = np.zeros([hsvImage.shape[0], hsvImage.shape[1], 3],dtype = 'uint8')
        for i in range(hsvImage.shape[0]):
            for j in range(hsvImage.shape[1]):
                minmax_img[i,j,2] = 255*(hsvImage[i,j,2]-min)/(max-min)

        rgbimg = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2RGB)
        return rgbimg

    def lutContrastEnhancement(self,image):
        xp = [0, 64, 128, 192, 255]
        fp = [0, 16, 128, 240, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        image = cv2.LUT(image, table)
        return image

    def histogramEqualisationOnVChannel(self,image):
        hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsvImage[:, :, 2] = cv2.equalizeHist(hsvImage[:, :, 2])
        equalizedImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2RGB)
        return equalizedImage

    def histogramEqalisation(self,image):
        bgrImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ycrcbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2YCrCb)
        ycrcbImage[:, :, 0] = cv2.equalizeHist(ycrcbImage[:, :, 0])
        equalizedImage = cv2.cvtColor(ycrcbImage, cv2.COLOR_YCrCb2BGR)
        return equalizedImage

    def showColorHistogramm(self,image, fileName):
        colors = ("red", "green", "blue")
        plt.figure()
        plt.xlim([0, 256])
        plt.title("Color Histogram")
        plt.xlabel("Color value")
        plt.ylabel("Pixel count")
        for channel_id, color in enumerate(colors):
            histogram, bin_edges = np.histogram(
                image[:, :, channel_id], bins=256, range=(0, 256)
            )
            plt.plot(bin_edges[0:-1], histogram, color=color)

       
        plt.show()
        plt.savefig(fileName)
        plt.clf()
        
        return histogram
    
    def showGreyScaleHistogramm(self,image,fileName):
        histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("grayscale value")
        plt.ylabel("pixel count")
        plt.xlim([0.0, 1.0])  # <- named arguments do not work here

        plt.plot(bin_edges[0:-1], histogram)
        plt.show()
        plt.savefig(fileName)
        plt.clf()
        return histogram
