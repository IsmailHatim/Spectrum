import torch

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries


def predict_fn(images, model):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = torch.stack([transforms.ToTensor()(img) for img in images]).to(device)
    
    with torch.no_grad():
        output = model(images) 
        output = output.squeeze(1) 

    proba = output.cpu().numpy()

    return np.stack([1 - proba, proba], axis=1) 


def show_lime(image_tensor, model):
    
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
    
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image_np,
        classifier_fn=lambda x: predict_fn(x, model),
        top_labels=1,
        num_samples=100
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=1
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(temp, mask))
    plt.axis('off')
    plt.title('Explication LIME')
    plt.show()