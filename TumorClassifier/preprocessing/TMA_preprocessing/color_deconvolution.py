
import fiona
import rasterio

from osgeo import gdal

import matplotlib.pyplot as plt
import numpy as np

import cv2

import argparse

import os


outPath = r"C:\AI\TMA\output"

ds = gdal.Open("C:\AI\TMA\cores\A-1.tif")


band1 = ds.GetRasterBand(1) # Red channel 
band2 = ds.GetRasterBand(2) # Green channel 
band3 = ds.GetRasterBand(3)


b1 = band1.ReadAsArray() 
b2 = band2.ReadAsArray() 
b3 = band3.ReadAsArray() 


print(b3.shape)

img = np.dstack((b1, b2, b3)) 

redPath = os.path.join(outPath,"red.png")
greenPath = os.path.join(outPath,"green.png")
bluePath = os.path.join(outPath,"blue.png")


redPathG = os.path.join(outPath,"redG.png")
greenPathG = os.path.join(outPath,"greenG.png")
bluePathG = os.path.join(outPath,"blueG.png")
otsuPath = os.path.join(outPath,"otsu.png") 
openPath = os.path.join(outPath,"open.png") 


	
#grayR = cv2.cvtColor(b1, cv2.COLOR_BGR2RGB)
#grayG = cv2.cvtColor(b2, cv2.COLOR_BGR2RGB)
#grayB = cv2.cvtColor(b3, cv2.COLOR_BGR2RGB)





ret, thresh1 = cv2.threshold(b3, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))

opening_img = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE , kernel )
                                            

#ret,thr = cv2.threshold(grayB, 0, 255, cv2.THRESH_OTSU)

#plt.imsave(redPathG,grayR)
#plt.imsave(greenPathG,grayG)
#plt.imsave(bluePathG,grayB)
cv2.imwrite(openPath,opening_img)




plt.imsave(redPath,b1)
plt.imsave(greenPath,b2)
plt.imsave(bluePath,b3)

