import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from captum.attr import IntegratedGradients

def show_integrated_gradient(model, image: torch.Tensor, device, threshold, baseline_type="black", n_steps=50) -> None:
    """Shows Integrated Gradients heatmap for the prediction of an input image using a ResNet model.

    Parameters
    ----------
    model : torch.nn.Module
        The trained ResNet model.
    image : torch.Tensor
        Image used for Integrated Gradients.
    device : torch.device
        Device (CPU/GPU) on which the model is running.
    baseline_type : str, optional
        Type of baseline image ("black" for a black image, "random" for a random noise image).
    n_steps : int, optional
        Number of interpolation steps for Integrated Gradients.

    Returns
    -------
    None
    """
    
    model.eval()
    image = image.to(device)
    
    # Define the baseline image (black or random noise)
    if baseline_type == "black":
        baseline = torch.zeros_like(image).to(device)
    elif baseline_type == "random":
        baseline = torch.randn_like(image).to(device)
    else:
        raise ValueError("Invalid baseline_type. Choose 'black' or 'random'.")

    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)

    # Compute attributions
    attributions = ig.attribute(image, baseline, target=0, n_steps=n_steps)
    
    # Convert attributions to numpy array for visualization
    attributions = attributions.squeeze().detach().cpu().numpy()
    attributions = np.abs(attributions).mean(axis=0)  # Average over RGB channels
    
    # Normalize attributions for visualization
    attributions -= attributions.min()
    attributions /= attributions.max()

    # Resize to match the input image
    gradcam_resized = cv2.resize(attributions, (image.shape[2], image.shape[3]))
    gradcam_thresholded = gradcam_resized > threshold

    return gradcam_resized, gradcam_thresholded
