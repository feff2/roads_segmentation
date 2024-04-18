import random
import time

import cv2
import pandas as pd
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from tqdm import trange, tqdm
from PIL import Image

from yaml_reader import yaml_reader

config = yaml_reader()

root_dir = config['dataset_param']['root_dir']
metadata_path = config['dataset_param']['metadata_path']
label_class_dict_path = config['dataset_param']['label_class_dict_path']


in_channels = config['model_params']['in_channels']
classes = config['model_params']['classes']

random_seed = config['training_params']['random_seed']
batch_size = config['training_params']['batch_size']
base_lr = config['training_params']
num_epochs = config['training_params']
weight_decay = config['training_params']


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'