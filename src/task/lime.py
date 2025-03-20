import torch

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries

def predict_fn(images):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = torch.stack([transforms.ToTensor()(img) for img in images]).to(device)
    
    with torch.no_grad():
        output = model(images) 
        output = output.squeeze(1) 

    proba = output.cpu().numpy()

    return np.stack([1 - proba, proba], axis=1) 