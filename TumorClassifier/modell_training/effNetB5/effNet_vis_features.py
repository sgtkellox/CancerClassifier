import torch
import cv2
import numpy as np
import glob as glob
import os
import urllib
from PIL import Image
from torchvision import transforms


from effNet_model import build_model
from torchvision import transforms

import matplotlib.pyplot as plt

class_names = ['Astro','GBM', 'Oligo']
DEVICE = 'cuda'

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook







if __name__ == '__main__':

    # Load the trained model.
    model = build_model(pretrained=True, fine_tune=True, num_classes=3)
    checkpoint = torch.load(r"C:\Users\felix\Desktop\models\model_14_pretrained.pth", map_location=DEVICE)
    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)

    print(model)


    model.eval()

    model_children = list(model.children())
    model_children[0][0].register_forward_hook(get_activation('fin'))

    filename = r"C:\Users\felix\Desktop\kryo\test\Oligo\O-N17-1080-K_52000_33500.jpg"
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    print(np.shape(activation['fin']))

    dimensi_iter = activation['fin'][0,:,:,:].size(dim=0)
    for i in range(dimensi_iter):
        print('processing layer %s'%(i+1))
        im2 = plt.imshow(activation['fin'][0,i,:,:].cpu().numpy(), cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear')
        plt.colorbar()
        plt.savefig(r'C:\Users\felix\Desktop\featureMap\fin_00_%s.png'%(i))
        plt.close()
    

 



