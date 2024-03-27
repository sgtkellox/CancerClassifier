import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import os
# batch size
BATCH_SIZE = 128

dataPath = r"/mnt/projects/neuropath_hd/data/splits/smear/other/smear/"

# the training transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomVerticalFlip(p=0.5),
    #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
    transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])
# the validation transforms
valid_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

# training dataset
train_dataset = datasets.ImageFolder(
    root=os.path.join(dataPath,"train"),
    transform=train_transform
)
# validation dataset
valid_dataset = datasets.ImageFolder(
    root=os.path.join(dataPath,"val"),
    transform=valid_transform
)
# training data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=8, pin_memory=True
)
# validation data loaders
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=8, pin_memory=True
)

