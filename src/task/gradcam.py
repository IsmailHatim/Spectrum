import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

def show_gradcam(model, image: torch.Tensor, device, threshold=0.5, conv_layer_index=-2):
    """Shows Grad-CAM heatmap for the prediction of an input image using the last conv layer of DenseNet.

    Parameters
    ----------
    image : torch.Tensor
        Image used for Grad-CAM.
    model : torch.nn.Module
        The trained DenseNet model.
    device : torch.device
        Device (CPU/GPU) on which the model is running.
    threshold : float, optional
        Threshold for binarizing the Grad-CAM heatmap.
    conv_layer_index : int, optional
        Index of the convolutional layer to analyze.

    Returns
    -------
    gradcam_resized : np.ndarray
        Grad-CAM heatmap.
    gradcam_thresholded : np.ndarray
        Thresholded Grad-CAM heatmap.
    """
    
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.clone())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].clone())

    layer = model.features[conv_layer_index]

    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)

    model.eval()
    image = image.to(device)
    output = model(image)

    target = output[0, 0]

    model.zero_grad()
    gradients.clear()
    target.backward()

    # Get activation maps & gradients
    act = activations[0].detach()
    grad = gradients[0].detach()

    # Compute weights for Grad-CAM
    weights = grad.mean(dim=[2, 3], keepdim=True)
    gradcam = F.relu((weights * act).sum(dim=1)).squeeze(0)
    
    # Normalize Grad-CAM
    gradcam -= gradcam.min()
    gradcam /= gradcam.max()
    
    gradcam_resized = cv2.resize(gradcam.cpu().numpy(), (image.shape[2], image.shape[3]))
    gradcam_thresholded = gradcam_resized > threshold

    return gradcam_resized, gradcam_thresholded