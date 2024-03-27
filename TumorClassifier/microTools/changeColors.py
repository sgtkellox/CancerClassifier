import cv2

import numpy as np





if __name__ == '__main__':
    
    imPath = r"C:\Users\felix\Desktop\snips\mapCut.png"
    
    imageName = imPath.split(".")[0]
    
    img = cv2.imread(imPath)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    img_copy = img.copy()
    
    
   
    
    
    img_copy[np.all(img_copy == (45, 35, 247), axis=-1)] = (245, 185, 66)
    img_copy[np.all(img_copy == (189, 255, 67), axis=-1)] = (0,0,255)
 
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                
    
    cv2.imwrite(imageName + "_changed.jpg", img_copy)
            
 
    img_width= img_copy.shape[1]
    img_height= img_copy.shape[0]
    
    for x in range(img_width):
        for y in range(img_height):
            
            print(img_copy[y][x])
                
    
    cv2.imwrite(imageName + "_changed.jpg", img_copy)
            
                
        