import numpy as np
#import cv2
import torch
import glob as glob
from torchvision import transforms
from torch.nn import functional as F
from torch import topk

import os 

from effNet_model import build_model


from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp

device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = build_model(pretrained=True, fine_tune=True, num_classes=3)

checkpoint = torch.load(r"C:\Users\felix\Desktop\neuro\models\model_14_pretrained.pth", map_location=device)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.to(device)
# Get your input
img = read_image(r"C:\Users\felix\Desktop\AutoEncoder\good\A-N17-1721-K_129000_45000.jpg")
# Preprocess it for your chosen model
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

with SmoothGradCAMpp(model) as cam_extractor:
  # Preprocess your data and feed it to the model
  out = model(input_tensor.unsqueeze(0))
  # Retrieve the CAM by passing the class index and the model output
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)




class_names = ['Astro','GBM', 'Oligo']

import matplotlib.pyplot as plt
# Visualize the raw CAM
plt.imshow(activation_map[0].squeeze(0).numpy())
plt.axis('off'); plt.tight_layout() 
plt.show()









  


    
    
    
    

        
       