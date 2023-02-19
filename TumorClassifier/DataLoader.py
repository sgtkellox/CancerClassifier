
import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split

project_name='BrainCancerClassifier'


import matplotlib.pyplot as plt
from WSI_handling import wsi


img_fname=r'D:\Ramin_SS_Oligo_Astro\A2-N17-1152K.svs'
xml_fname=r'D:\Ramin_SS_Oligo_Astro\A2-N17-1152K.xml'
w = wsi(img_fname,xml_fname)

plt.imshow(w.get_wsi(desired_mpp=8));

plt.imshow(w.mask_out_annotation(desired_mpp=8));




def sortImagesByFileName(path):

    return