import os
import shutil
import cv2
import argparse




def aboveThreshOld(tile,destPath, image):
    h, w, _ = tile.shape
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    pixels = cv2.countNonZero(thresh)
    ratio = (pixels/(h * w)) * 100

    

    if ratio < 50:
        image = image.split(".")[0]
        image = image + "mask.jpg"

    
        safePath = os.path.join(destPath,image)

        cv2.imwrite(safePath,thresh)
        return False
    else:
        return True



if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-i", "--input", help="The path to the folder containing the slides")
    argParser.add_argument("-o", "--out", help="The path to the folder where u want the tiles")
   

    args = argParser.parse_args()

    pathIn = args.input
  
    pathOut = args.out

    for image in os.listdir(pathIn):
        srcPath = os.path.join(pathIn,image)
        img = cv2.imread(srcPath)
        if not aboveThreshOld(img,pathOut,image):
            
            destPath = os.path.join(pathOut,image)
            shutil.move(srcPath,destPath)
