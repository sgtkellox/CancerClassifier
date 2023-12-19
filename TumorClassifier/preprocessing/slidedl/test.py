import glob
import os
import pickle
import sys
from pathlib import Path
import random
from tqdm import tqdm

#sys.path.append('/path/to/slidl') # not necessary if you install SliDL with pip
#sys.path.append(r"C:\AI\slidl-tutorial\Pytorch-UNet")

add_dll_dir = getattr(os, "add_dll_directory", None)
vipsbin = r"C:\AI\vips-dev-8.14\bin"
if callable(add_dll_dir):
    add_dll_dir(vipsbin)
    print("added dir")
else:
    os.environ["PATH"] = os.pathsep.join((vipsbin, os.environ["PATH"]))
    print("added path")
    

from slidl.slide import Slide

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import albumentations as A
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import time
import logging