

from torch import nn

import timm


import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm


class CFG:

  epochs =20                              # No. of epochs for training the model
  lr = 0.001                              # Learning rate
  batch_size = 16                         # Batch Size for Dataset

  model_name = 'tf_efficientnet_b4_ns'    # Model name (we are going to import model from timm)
  img_size = 224                          # Resize all the images to be 224 by 224

  # going to be used for loading dataset
  train_path='/content/drive/MyDrive/chest_xray/train'
  validate_path='/content/drive/MyDrive/chest_xray/val'
  test_path='/content/drive/MyDrive/chest_xray/test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("On which device we are on:{}".format(device))









