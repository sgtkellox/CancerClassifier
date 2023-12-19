import os
from os import listdir
from PIL import Image

import argparse



def checkFiles(path):
   counter = 0
   for file in os.listdir(path):
      if file.endswith('.jpg'):
         
         filePath = os.path.join(path,file)
         try:             
            im = Image.load(filePath)
            im.verify() #I perform also verify, don't know if he sees other types o defects
            im.close() #reload is necessary in my case
            im = Image.load(filePath) 
            im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            im.close()
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
   

