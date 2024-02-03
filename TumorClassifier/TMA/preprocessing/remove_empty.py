import os
import cv2
import numpy as np

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def threshOldRGB(image, threshold):
    
    return


def thresholdGrey(imagePath, threshold):
	for file in os.listdir(imagePath):
		imPath = os.path.join(imagePath,file)
		image = cv2.imread(imPath)
		
		avg = np.average(image)
		
		if avg>threshold:
			os.remove(imPath)
		
			print("image "+ file + " removed since avg is only " + str(avg))
	
     



    

if __name__ == "__main__":
	imagePath = r"D:\mix\grey"
	
	thresholdGrey(imagePath, 254.8)
		
		
    
    
    