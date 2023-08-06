import fiona
import rasterio
import rasterio.mask

import matplotlib.pyplot as plt
import numpy as np

import cv2

tilePath = r""

with fiona.open(r"C:\Users\felix\Desktop\GBM-N20-0322-K-Q0.geojson", "r") as geojson:
   features = [feature["geometry"] for feature in geojson]

with rasterio.open(r"F:\split\kryo\train\GBM\GBM-N20-0322-K-Q0.svs") as src:
   out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
   out_meta = src.meta



out_image = np.moveaxis(out_image, 0, -1)

x,y = 0

while x < out_image.shape[1]-500:
    while y < out_image.shape[0]-500:

        tile = out_image[y:y+500,x:x+500]
        if not [255,255,255] in tile:





print(out_image.shape)

plt.imshow(out_image)
plt.show()

