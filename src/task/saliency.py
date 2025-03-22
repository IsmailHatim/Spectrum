import torch

def show_saliency(model, image, device, threshold=0.5, conv_layer_index=None):
    """
    Generate a saliency map for a given model and input image.

    Parameters
    ----------
    model : torch.nn.Module
        The trained CNN model.
    image : torch.Tensor
        Image used for saliency map. Shape: (1, C, H, W)
    device : torch.device
        Device (CPU/GPU) on which the model is running.
    threshold : float, optional
        Threshold for binarizing the saliency map.
    conv_layer_index : int, optional
        Not used for saliency, present for compatibility.

    Returns
    -------
    saliency : torch.Tensor
        Normalized saliency map. Shape: (H, W)
    saliency_thresholded : torch.Tensor
        Thresholded binary saliency map. Shape: (H, W)
    """

    # Ensure model is in evaluation mode
    model.eval()

    # Clone image to avoid in-place modifications and enable gradients
    input_img = image.clone().detach().to(device)
    input_img.requires_grad_(True)

    # Forward pass
    output = model(input_img)

    # Take the score of the predicted class (max logit)
    score, _ = torch.max(output, dim=1)
    score = score.sum()  # In case of batch

    # Backward pass to get gradients w.r.t input
    model.zero_grad()
    score.backward()

    # Compute the saliency map: max over channels of absolute gradient
    saliency, _ = torch.max(input_img.grad.data.abs(), dim=1)
    saliency = saliency.squeeze()  # Shape: (H, W)

    # Normalize the saliency map
    saliency -= saliency.min()
    saliency /= (saliency.max() + 1e-8)  # Avoid division by zero

    # Threshold the map
    saliency_thresholded = saliency > threshold

    return saliency.cpu(), saliency_thresholded.cpu()
