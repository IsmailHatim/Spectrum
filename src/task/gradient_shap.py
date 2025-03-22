import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_gradient_shap(model, image: torch.Tensor, device, threshold=0.5, num_samples=50, conv_layer_index=None):
    """Shows GradientSHAP heatmap for the prediction of an input image.

    Parameters
    ----------
    model : torch.nn.Module
        The trained CNN model.
    image : torch.Tensor
        Image used for GradientSHAP.
    device : torch.device
        Device (CPU/GPU) on which the model is running.
    threshold : float, optional
        Threshold for binarizing the GradientSHAP heatmap.
    num_samples : int, optional
        Number of samples for stochastic perturbations (default is 50).

    Returns
    -------
    None
    """
    
    model.eval()
    image = image.to(device)

    # Use GradientSHAP explainer
    explainer = shap.GradientExplainer(model, torch.randn(1, 3, 256, 256).to(device))

    # Generated SHAP values with
    shap_values = explainer.shap_values(image, nsamples=num_samples)

    # Convert SHAP values into aggregated heatmap
    shap_values = np.array(shap_values).squeeze()
    shap_heatmap = np.abs(shap_values).mean(axis=0)  # Moyenne sur les canaux RGB

    # Normalise heatmap
    shap_heatmap -= shap_heatmap.min()
    shap_heatmap /= shap_heatmap.max()

    shap_heatmap_resized = cv2.resize(shap_heatmap, (image.shape[2], image.shape[3]))
    shap_heatmap_thresholded = shap_heatmap_resized > threshold

    return shap_heatmap_resized, shap_heatmap_thresholded