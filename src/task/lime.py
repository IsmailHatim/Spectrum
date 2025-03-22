import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from skimage.segmentation import mark_boundaries
from PIL import Image

from lime.lime_image import LimeImageExplainer


def show_lime(model, image: torch.Tensor, device=None, threshold=None) -> None:
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

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    def predict_func(images):
        
        tensor_images = torch.stack([transform(Image.fromarray((img * 255).astype(np.uint8))) for img in images])
        with torch.no_grad():
            outputs = model(tensor_images)
            print(outputs)
            probs = torch.sigmoid(outputs).numpy()

        return probs


    explainer = LimeImageExplainer()

    explanation = explainer.explain_instance(image_np, 
                                             predict_func, 
                                             top_labels=1, 
                                             num_samples=100)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                positive_only=True, 
                                                num_features=1, 
                                                hide_rest=False)
    
    # Plot the original and LIME-processed images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_np)
    ax[0].set_title("Input Image")
    
    ax[1].imshow(mark_boundaries(temp, mask))
    ax[1].set_title("LIME Explanation")
    
    plt.show()
