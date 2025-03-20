import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries, quickshift, slic, felzenszwalb

def show_lime(model, image: torch.Tensor, device, num_samples=100, num_features = 30) -> None:
    """
    Shows LIME heatmap for the prediction of an input image using a modified ResNet.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor used for LIME analysis.
    model : torch.nn.Module
        The trained ResNet model.
    device : torch.device
        Device (CPU/GPU) on which the model is running.

    Returns
    -------
    None
    """
    model.eval()
    image_np = image.cpu().numpy().squeeze().transpose(1, 2, 0)  # Convert to HxWxC format
    
    # Define a wrapper function for LIME that takes numpy arrays
    def model_predict(images_np):
        images_tensors = torch.tensor(images_np, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        
        # Adding gaussian noise
        noise = torch.randn_like(images_tensors) * 0.05  # Ajuste l'intensité si nécessaire
        images_tensors += noise
        
        outputs = model(images_tensors).detach().cpu().numpy()
        
        return np.hstack([1 - outputs, outputs])
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np,
        model_predict,
        top_labels=1,
        hide_color=(0,0,0),
        num_samples=num_samples,  # Number of perturbations
        segmentation_fn=quickshift
    )
    
    # Get the LIME mask
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=num_features,  # Number of superpixels to highlight
        hide_rest=False
    )
    
    # Plot the original and LIME-processed images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_np)
    ax[0].set_title("Input Image")
    
    ax[1].imshow(mark_boundaries(temp, mask))
    ax[1].set_title("LIME Explanation")
    
    plt.show()
