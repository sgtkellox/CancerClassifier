from stardist.models import StarDist2D

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois

# prints a list of available models
StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_he')

lbl_cmap = random_label_cmap()


from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt

img = plt.imread(r"C:\Users\felix\Desktop\neuro\newTilingTest\th235\tiles\A2-N20-0874-K-Q2_40000_8500.jpg")

labels, details = model.predict_instances(normalize(img))

for entry in details:
    print(len(details['points']))




plt.figure(figsize=(8,8))
plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
plt.axis('off')
plt.show()
