import torch
import cv2
import glob as glob
import os
from model import CustomCNN
from torchvision import transforms
# Constants.
path = r'C:\Users\felix\Desktop\neuro\kryo\val\Astro'

classes = ["artefact","good"]

IMAGE_SIZE = 224
device = 'cuda'
# Load the trained model.
model = CustomCNN(num_classes=1)
checkpoint = torch.load(r'C:\Users\felix\Desktop\AutoEncoder\models\model73.pth', map_location=device)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ]) 


model = model.to(device)


for image_name in os.listdir(path):

    imagePath = os.path.join(path,image_name)
    
    # Read the image and create a copy.
    image = cv2.imread(imagePath)
    orig_image = image.copy()
    
    # Preprocess the image.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    
    # Forward pass throught the image.
    outputs = model(image)
    output_sigmoid = torch.sigmoid(outputs)
    pred_class_name = 'artifact' if output_sigmoid > 0.5 else 'normal'

    if pred_class_name == "artifact":
        safePath = os.path.join(r"C:\Users\felix\Desktop\AutoEncoder\results\artefact",image_name)     
    else:
        safePath = os.path.join(r"C:\Users\felix\Desktop\AutoEncoder\results\good",image_name)
    cv2.imwrite(safePath,orig_image)


    
   
    