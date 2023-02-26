import slideio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import ContrastEnhancmentUtils 

class SlideSegmenter:
    
    def __init__(self):
        pathToOutputFolder = ""

        

    def isBackGround(self,image):
        mean = np.mean(image, axis=(0, 1, 2))

        if mean > 100 and mean < 200:
            return False
        else:
           return True


    def makeSlices(self, sizeOfSlice, scene):
        wsiWidth = scene.rect[2]
        wsiHeight = scene.rect[3]
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
                    outpathHistogram = r"C:\Users\felix\Desktop\croppedImages2\crop_histogram" + str(count) + ".jpg"
                    cv2.imwrite(outpath, image)
                    ceu.showColorHistogramm(image,outpathHistogram)

                    #outpathEnhandced = r"C:\Users\felix\Desktop\ContrastEnhancment\enhancedCrop" + str(count) + ".jpg"
                    #cv2.imwrite(outpathEnhandced, contrastEnhancedImage)

                    count +=1
    
        return

    def makeSlicesWithOverlap(self,sizeOfSlice,overlap, scene):
        contrastTools = ContrastEnhancmentUtils.ContrastEnhancmentUtils()

        wsiWidth = scene.rect[2]
        wsiHeight = scene.rect[3]
        minX = 0
        minY = 0
        count = 1;
        while minX+sizeOfSlice-overlap < wsiWidth:
            while minY+sizeOfSlice-overlap < wsiHeight:
            
                image = scene.read_block((minX,minY,sizeOfSlice,sizeOfSlice), (sizeOfSlice,sizeOfSlice))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                minY += (sizeOfSlice-overlap)

                if not self.isBackGround(image):
                    
                    contrastEnhancedImage = contrastTools.histogramEqalisation(image)
                    outpath = r"C:\Users\felix\Desktop\croppedImages\crop" + str(count) + ".jpg"
                    outpathHistogram = r"C:\Users\felix\Desktop\croppedImages2\crop_histogram" + str(count) + ".jpg"
                    cv2.imwrite(outpath, image)
                    contrastTools.showColorHistogramm(contrastEnhancedImage,outpathHistogram)
                    #contrastEnhancedImage = histogramEqalisation(image)
                    #outpathEnhandced = r"C:\Users\felix\Desktop\ContrastEnhancment\enhancedCrop" + str(count) + ".jpg"
                    #cv2.imwrite(outpathEnhandced, contrastEnhancedImage)
                    if count>1:
                        return
                    count +=1
            minX += (sizeOfSlice-overlap)
            minY = 0
        return


    def makeSlicesWithOverlapNoFilter(self,sizeOfSlice,overlap, scene):
        contrastTools = ContrastEnhancmentUtils.ContrastEnhancmentUtils()

        wsiWidth = scene.rect[2]
        wsiHeight = scene.rect[3]
        minX = 0
        minY = 0
        count = 1;
        while minX+sizeOfSlice-overlap < wsiWidth:
            while minY+sizeOfSlice-overlap < wsiHeight:
            
                image = scene.read_block((minX,minY,sizeOfSlice,sizeOfSlice), (sizeOfSlice,sizeOfSlice))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                minY += (sizeOfSlice-overlap)

                #contrastEnhancedImage = contrastTools.histogramEqalisation(image)
                outpath = r"C:\Users\felix\Desktop\eindri\tiles\tile" + str(count) + ".jpg"
                #outpathHistogram = r"C:\Users\felix\Desktop\croppedImages2\crop_histogram" + str(count) + ".jpg"
                cv2.imwrite(outpath, image)
                #contrastTools.showColorHistogramm(contrastEnhancedImage,outpathHistogram)
                #contrastEnhancedImage = histogramEqalisation(image)
                #outpathEnhandced = r"C:\Users\felix\Desktop\eindri\contrastEnhancedTiles" + str(count) + ".jpg"
                #cv2.imwrite(outpathEnhandced, contrastEnhancedImage)
                    
                count +=1
                    
                    
            minX += (sizeOfSlice-overlap)
            minY = 0
        return
















