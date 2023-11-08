import fiona
import rasterio
import rasterio.mask

import matplotlib.pyplot as plt
import numpy as np



import os


slidePath= r"F:slides\kryoQ2\GBM-N17-1629-K-Q2.svs"

#tilePath = r"C:\Users\felix\Desktop\AutoEncoder\gtTiles"

with fiona.open(r"C:\Users\felix\Desktop\FoldJsons\GBM-N17-1629-K-Q2.geojson", "r") as geojson:
   features = [feature["geometry"] for feature in geojson]


with rasterio.open(slidePath) as src:
   out_image, out_transform = rasterio.mask.mask(src, features, crop=False)
   out_meta = src.meta
   out_image = np.moveaxis(out_image, 0, -1)
   #array = src.read(1)
   
   #mask = rasterio.features.geometry_mask(features,array.shape,transform=src.transform )
   

   #plt.imshow(mask)
   #plt.show()

   imageName = slidePath.split("\\")[-1].split(".")[0]

   




   plt.imshow(out_image)
   plt.show()


