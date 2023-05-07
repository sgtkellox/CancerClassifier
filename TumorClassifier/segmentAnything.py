from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt



sam = sam_model_registry["vit_l"](checkpoint=r"C:\Users\felix\Downloads\sam_vit_l_0b3195.pth")
device = "cuda"
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
image = cv2.imread(r"C:\Users\felix\Desktop\Neuro\AugmentOutput\greyAfterHistEq\GBM-N21-217Q_58001_34001.jpg")

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
                
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)
plt.figure(figsize=(10,10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 

