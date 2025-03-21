import torch
import matplotlib.pyplot as plt


def show_saliency(model, image, device, threshold):

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