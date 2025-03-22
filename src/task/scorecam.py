import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def show_scorecam(model, image: torch.Tensor, device, threshold, conv_layer_index = -2, num_samples=100) -> None:
    """Optimized Score-CAM heatmap for a binary classification model (defective vs. not defective).

    Parameters
    ----------
    model : torch.nn.Module
        The trained CNN model.
    image : torch.Tensor
        Image used for Score-CAM.
    conv_layer_index : int
        Index of the convolutional layer to analyze.
    device : torch.device
        Device (CPU/GPU) on which the model is running.
    num_samples : int, optional
        Number of perturbations for Score-CAM estimation (default is 100).

    Returns
    -------
    None
    """

    # Move image to device
    image = image.to(device, non_blocking=True)

    # Function to store activations
    activation = None

    def forward_hook(module, input, output):
        nonlocal activation
        activation = output.detach()

    # Hook into the target convolutional layer
    if model._get_name() == "DenseNet":
        layer = model.features[conv_layer_index]
    elif model._get_name() == "ResNet":
        layer = model.layer4[conv_layer_index]
    else:
        raise ValueError("Model not supported. Choose DenseNet or ResNet.")

    hook = layer.register_forward_hook(forward_hook)

    model.eval()

    # Forward pass to get model prediction
    with torch.no_grad():
        output = model(image)  # No sigmoid, since it's already applied inside the model

    # Remove hook to free memory
    hook.remove()

    # Get activations and free memory
    act = activation
    activation = None  # Free memory
    torch.cuda.empty_cache()

    # Rescale activations to [0, 1]
    act -= act.min()
    act /= act.max()

    # Get shape info
    C, H, W = act.shape[1], act.shape[2], act.shape[3]

    # Initialize importance scores
    scores = torch.zeros(C, device=device)

    # Iterate over activation channels one by one (to reduce memory consumption)
    for i in range(C):
        with torch.no_grad():
            activation_map = act[:, i:i+1, :, :]  # Extract a single channel
            activation_resized = F.interpolate(activation_map, size=(image.shape[2], image.shape[3]), mode='bilinear')

            # Multiply original image with activation map
            weighted_image = image * activation_resized

            # Forward pass with masked image
            output_masked = model(weighted_image)
            scores[i] = output_masked[0]

        # Free memory
        del activation_map, activation_resized, weighted_image, output_masked
        torch.cuda.empty_cache()

    # Normalize scores
    scores -= scores.min()
    scores /= scores.max()

    # Ensure scores shape is broadcastable
    scores = scores.view(C, 1, 1)  # Shape: [C, 1, 1]

    # Compute final Score-CAM heatmap (channel-wise weighted sum)
    score_cam = (scores * act[0]).sum(dim=0)  # Shape: [H, W]

    # Normalize Score-CAM
    score_cam -= score_cam.min()
    score_cam /= score_cam.max()

    # Resize final heatmap to input image size
    score_cam_resized = cv2.resize(score_cam.cpu().numpy(), (image.shape[3], image.shape[2]))
    score_cam_thresholded = score_cam_resized > threshold

    # Free memory
    del act, scores, score_cam
    torch.cuda.empty_cache()

    return score_cam_resized, score_cam_thresholded
