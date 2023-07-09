import torch
import cv2
import numpy as np
import glob as glob
import os

from effNet_model import build_model
from torchvision import transforms

# Constants.
DATA_PATH = r'C:\Users\felix\Desktop\kryo\test\GBM'
IMAGE_SIZE = 224
DEVICE = 'cuda'

# Class names.
class_names = ['Astro', 'GBM', 'Oligo']

# Load the trained model.
model = build_model(pretrained=True, fine_tune=True, num_classes=3)
checkpoint = torch.load(r"C:\Users\felix\Desktop\models\model_16_pretrained.pth", map_location=DEVICE)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])

model.to(DEVICE)
all_image_paths = glob.glob(f"{DATA_PATH}/*")
model.eval()
# Iterate over all the images and do forward pass.
for image_path in all_image_paths:
    # Get the ground truth class name from the image path.
    split = image_path.split(os.path.sep)
    
    gt_class_name = split[-2]
    imageName = split[-1].split(".")[0]
    print(imageName)
    
    # Read the image and create a copy.
    image = cv2.imread(image_path)
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
    print(f"GT: {gt_class_name}, Pred: {pred_class_name.lower()}")
    # Annotate the image with ground truth.
    cv2.putText(
        orig_image, f"GT: {gt_class_name}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA
    )
    # Annotate the image with prediction.
    cv2.putText(
        orig_image, f"Pred: {pred_class_name.lower()}",
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (100, 100, 225), 2, lineType=cv2.LINE_AA
    ) 
   
    safepath = os.path.join(r"C:\Users\felix\Desktop\outPut",imageName+".jpg")
    cv2.imwrite(safepath, orig_image)