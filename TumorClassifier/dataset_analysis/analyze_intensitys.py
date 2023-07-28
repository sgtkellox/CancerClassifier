import os 
import argparse

import cv2

import numpy as np





    
    

def collectDatasetStatistics(folder):

    avgs = []

    for imageName in os.listdir(folder):

         imagePath = os.path.join(path,imageName)
         image = cv2.imread(imagePath)
         orig_image = image.copy()
    
    
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

         avg = np.average(image)
         if avg > 235:
             print(imageName)
         avgs.append(avg)

    maxVal = max(avgs)
    minVal = min(avgs)

    return minVal, maxVal

         






if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-p", "--path", help="path containing tiles")
    

    

    args = argParser.parse_args()

    path = args.path

    path = r"C:\Users\felix\Desktop\neuro\thTestSmear\95_100\tiles"

    minVal , maxVal = collectDatasetStatistics(path)

    print("min: "+ str(minVal))
    print("\n")

    print("max: "+ str(maxVal))




  

   
    
    
  
   