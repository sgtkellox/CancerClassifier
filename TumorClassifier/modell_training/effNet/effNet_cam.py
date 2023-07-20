import numpy as np
import cv2
import torch
import glob as glob
from torchvision import transforms
from torch.nn import functional as F
from torch import topk

import os 

from effNet_model import build_model

#from grad_cam import GradCAM
DEVICE = 'cuda'

class_names = ['Astro','GBM', 'Oligo']

features_blobs = []

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225])
    ])


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam



def show_cam(CAMs, width, height, orig_image, class_idx, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + orig_image * 0.5
        # put class label text on the result
        cv2.putText(result, class_names[int(class_idx[i])], (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(save_name, result)
        
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

if __name__ == "__main__":

    imageFolderPath = r"C:\Users\felix\Desktop\neuroImages\kryo\test\Astro"

    outpath = r"C:\Users\felix\Desktop\effNetB0Cams\Astro"

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
# initialize model, switch to eval model, load trained weights
    model = build_model(pretrained=True, fine_tune=True, num_classes=3)
    checkpoint = torch.load(r"C:\Users\felix\Desktop\models\model_14_pretrained.pth", map_location=DEVICE)
    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(DEVICE)

    print(model)


    
    
    model._modules.get('features').register_forward_hook(hook_feature)
    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    for imageName in os.listdir(imageFolderPath):
    # read the image
        imageNameFull = os.path.join(imageFolderPath,imageName)
        image = cv2.imread(imageNameFull)
        orig_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = np.expand_dims(image, axis=2)
        height, width, _ = orig_image.shape
        # apply the image transforms
        image_tensor = transform(image)
        image_tensor.to(DEVICE)
        # add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(DEVICE)
        
        # forward pass through model
        outputs = model(image_tensor)
        # get the softmax probabilities
        probs = F.softmax(outputs).data.squeeze()
        # get the class indices of top k probabilities
        class_idx = topk(probs, 1)[1].int()
        className = class_names[class_idx]
    
        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
        # file name to save the resulting CAM image with
        save_name = os.path.join(outpath, imageName)
        #print(save_name)
        # show and save the results
        show_cam(CAMs, width, height, orig_image, class_idx, save_name)


    