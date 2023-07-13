import torch
import cv2
import numpy as np
import glob as glob
import os
from effNet_model import build_model
from torchvision import transforms
# Constants.

IMAGE_SIZE = 224
DEVICE = 'cuda'
# Class names.
class_names = ['Astro','GBM', 'Oligo']
# Load the trained model.
model = build_model(pretrained=True, fine_tune=True, num_classes=3)
checkpoint = torch.load(r'C:\Users\felix\Desktop\neuro\models\model_15_pretrained.pth', map_location=DEVICE)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)

model.eval()

# Get all the test image paths.
path = r"C:\Users\felix\Desktop\neuro\kryo\test\Oligo"
# Iterate over all the images and do forward pass.

right = 0
wrong = 0
images = os.listdir(path)
for imgName in images:

    gt_class_name = path.split(os.path.sep)[-1]

    print(gt_class_name)
    # Read the image and create a copy.
    imgPath = os.path.join(path,imgName)
    image = cv2.imread(imgPath)
    orig_image = image.copy()
    
    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(DEVICE)
    
    # Forward pass throught the image.
    outputs = model(image)
    outputs = outputs.detach().cpu().numpy()
    pred_class_name = class_names[np.argmax(outputs[0])]
    if pred_class_name == gt_class_name:
        right +=1
    
    # Annotate the image with ground truth.

    acc = right/len(images)

    print("acc "+ str(acc))
   


    

