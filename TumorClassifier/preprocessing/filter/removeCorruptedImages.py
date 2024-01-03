import os

import cv2

import argparse
import shutil



def checkFiles(path):
   counter = 0
   for file in os.listdir(path):
      if file.endswith('.jpg'):
         
         filePath = os.path.join(path,file)
         try:             
            img = cv2.imread(filePath)
            
         except: 
              os.remove(filePath)
              counter+=1
      else:
          
          os.remove(filePath)
          counter+=1
   print("removed " + str(counter)+ " files") 

      
         

if __name__ == '__main__':
   
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-p", "--path", help="The path to the folder containing the images")
  
    
    args = argParser.parse_args()

    path = args.path
    
    
    checkFiles(path)
   

