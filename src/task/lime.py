import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from lime.lime_image import LimeImageExplainer


def show_lime(model, image: torch.Tensor, device, threshold=None, conv_layer_index=None):
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
    temp : np.ndarray
        LIME heatmap.
    mask : np.ndarray
        Mask of the LIME heatmap.
    """

    model.eval()
    image_np = image.cpu().numpy().squeeze().transpose(1, 2, 0)  # Convert to HxWxC format

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    def predict_func(images):
        
        tensor_images = torch.stack([transform(Image.fromarray((img * 255).astype(np.uint8))) for img in images])
        with torch.no_grad():
            outputs = model(tensor_images.to(device))
            probs = torch.sigmoid(outputs).cpu().numpy()

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
    
    return temp, mask
    
