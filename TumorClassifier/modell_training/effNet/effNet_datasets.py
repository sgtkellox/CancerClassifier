import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
# Required constants.
import os

import os 


IMAGE_SIZE = 224 # Image size of resize when applying transforms.
BATCH_SIZE = 256
NUM_WORKERS = 10 # Number of parallel processes for data preparation.

# Training transforms
def get_train_transform(IMAGE_SIZE, pretrained):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        #transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        #transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_transform
# Validation transforms
def get_valid_transform(IMAGE_SIZE, pretrained):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return valid_transform
# Image normalization transforms.
def normalize_transform(pretrained):
    if pretrained: # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
            )
    else: # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize

def get_datasets(pretrained):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along 
    with the class names.
    """
    

    train_dataset = datasets.ImageFolder(
        root=os.path.join(dataPath,"train"),
        transform=(get_train_transform(IMAGE_SIZE, pretrained))
    )
# validation dataset
    valid_dataset = datasets.ImageFolder(
        root=os.path.join(dataPath,"val"),
        transform=(get_valid_transform(IMAGE_SIZE, pretrained))
    )
    

    return train_dataset, valid_dataset, train_dataset.classes


def get_data_loaders(dataset_train, dataset_valid):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader 

