import fiona
import rasterio
import rasterio.mask

import matplotlib.pyplot as plt
import numpy as np

import cv2

import os

slidePath= r"C:\Users\felix\Desktop\fixedKryoTest\kryoTest\GBM-N18-3107-K-Q2.svs"

tilePath = r"C:\Users\felix\Desktop\goodjson\toCheck"

with fiona.open(r"C:\Users\felix\Desktop\goodjson\jsons\GBM-N18-3107-K-Q2.geojson", "r") as geojson:
   features = [feature["geometry"] for feature in geojson]

with rasterio.open(slidePath) as src:
   out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
   out_meta = src.meta

imageName = slidePath.split("\\")[-1].split(".")[0]

out_image = np.moveaxis(out_image, 0, -1)

x = 0

y = 0

print("x : "+ str(out_image.shape[1]))
print("y : "+ str(out_image.shape[0]))

while x < out_image.shape[1]-500:
    y = 0
    while y < out_image.shape[0]-500:

        print("X " +str(x))
        print("Y " +str(y))

        tile = out_image[y:y+500,x:x+500]

        #tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

        #number_of_black_pix = cv2.countNonZero(tile)
        
        number_of_black_pix  = np.sum(tile == (0,0,0))

        print(number_of_black_pix)

        if not number_of_black_pix>10000:
            safePath = os.path.join(tilePath,imageName+"_"+str(x)+"_"+str(y)+".jpg")
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            cv2.imwrite(safePath,tile)
        y+=500
    x+=500

plt.imshow(out_image)
plt.show()


#print(out_image.shape)



