import torch
import matplotlib.pyplot as plt


def show_saliency(model, image, device, threshold=0.5, conv_layer_index=-2):
    """Shows saliency map for the prediction of an input image."

    Parameters
    ----------
    model : torch.nn.Module
        The trained CNN model.
    image : torch.Tensor
        Image used for saliency map.
    device : torch.device
        Device (CPU/GPU) on which the model is running.
    threshold : float, optional
        Threshold for binarizing the saliency map.
    conv_layer_index : int, optional
        Index of the convolutional layer to analyze.

    Returns
    -------
    saliency : torch.Tensor
        Saliency map.
    saliency_thresholded : torch.Tensor
        Thresholded saliency map.
    """


    image = image.to(device)
    image.requires_grad = True

    output = model(image)

    model.zero_grad()  
    output.backward()

    saliency, _ = torch.max(image.grad.data.abs(), dim=1)

    saliency = saliency - saliency.min()
    saliency = saliency / saliency.max()
    
    saliency = saliency.cpu().squeeze()
    image = image.detach()
    
    saliency_thresholded = saliency > threshold

    return saliency, saliency_thresholded