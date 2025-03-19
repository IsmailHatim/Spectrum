import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_gradient_shap(model, image: torch.Tensor, device, num_samples=50) -> None:
    """Shows GradientSHAP heatmap for the prediction of an input image.

    Parameters
    ----------
    model : torch.nn.Module
        The trained CNN model.
    image : torch.Tensor
        Image used for GradientSHAP.
    device : torch.device
        Device (CPU/GPU) on which the model is running.
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

    # Generated SHAP values with noise
    shap_values = explainer.shap_values(image, nsamples=num_samples)

    # Convert SHAP values into aggregated heatmap
    shap_values = np.array(shap_values).squeeze()
    shap_heatmap = np.abs(shap_values).mean(axis=0)  # Moyenne sur les canaux RGB

    # Normalise heatmap
    shap_heatmap -= shap_heatmap.min()
    shap_heatmap /= shap_heatmap.max()

    shap_heatmap_resized = cv2.resize(shap_heatmap, (image.shape[2], image.shape[3]))

    # Plot the result
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image.cpu().squeeze().permute(1, 2, 0))
    ax[0].set_title("Input Image")

    ax[1].imshow(image.cpu().squeeze().permute(1, 2, 0))
    ax[1].imshow(shap_heatmap_resized, cmap='jet', alpha=0.5)
    ax[1].set_title(f"GradientSHAP Heatmap")

    plt.show()