import os
import PIL
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from torchsummary import summary
import torch
from torch.autograd import Variable
from torch import nn
import torchvision
from torchvision import transforms,datasets,models
import splitfolders
from monai.networks.nets import DenseNet201,DenseNet121
from sklearn.metrics import accuracy_score,classification_report

path = r""

test_path = os.path.join(path,'test')
train_path = os.path.join(path,'train')
val_path = os.path.join(path,'val')


data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-3,3)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                    ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                   ]),
        "test":transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
}

data_sets = {
    "train":datasets.ImageFolder(train_path,transform = data_transforms['train']),
    "val":datasets.ImageFolder(val_path,transform = data_transforms['val']),
    "test":datasets.ImageFolder(test_path,transform = data_transforms['test']),
}
num_train = len(data_sets["train"])
num_val = len(data_sets["val"])
datasizes = {'train':num_train,'val':num_val}

data_dicts = {
    'train':[],
    'test':[],
    'val':[]
}
data_labels = {
    'train':[],
    'test':[],
    'val':[]
}

for phase in ['train','val','test']:
    if phase == 'train':
        data_dir = train_path
    elif phase == 'val':
        data_dir = val_path
    else:
        data_dir = test_path
    class_names = sorted(x for x in os.listdir(data_dir)
                            if os.path.isdir(os.path.join(data_dir, x)))
    num_class = len(class_names) 
    image_files = [
        [
            os.path.join(data_dir, class_names[i], x)
            for x in os.listdir(os.path.join(data_dir, class_names[i]))
        ]
        for i in range(num_class)
    ]
    num_each = [len(image_files[i]) for i in range(num_class)]
    image_files_list = [] 
    image_class = []
    for i in range(num_class):
        image_files_list.extend(image_files[i])
        image_class.extend([i] * num_each[i])
    data_dicts[phase] = image_files_list
    data_labels[phase] = image_class
    num_total = len(image_class)
    image_width, image_height = PIL.Image.open(image_files_list[0]).size
    print(f"{phase} information")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")
    print(f"Total image count: {num_total}")
    print('--'*8)
    
df = pd.DataFrame({'path': data_dicts['train'],'label': data_labels['train']})
sorted_counts  = df.value_counts('label')
plt.title('distribution in train dataset')
plt.pie(sorted_counts, labels = class_names,startangle = 90,counterclock = False,autopct="%.1f%%")

batch_size = 32
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))

dataloaders = {
    "train":torch.utils.data.DataLoader(data_sets['train'],batch_size=batch_size,shuffle=True,num_workers=nw),
    "val":torch.utils.data.DataLoader(data_sets['val'],batch_size=batch_size,shuffle=False,num_workers=nw),
    "test":torch.utils.data.DataLoader(data_sets['test'],batch_size=batch_size,shuffle=False,num_workers=nw)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=models.mobilenet_v3_large(pretrained=True)
num_features=model.classifier[0].in_features
model.classifier=nn.Sequential(
    nn.Linear(in_features=num_features, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4, bias=True)
  )
model.to(device)




