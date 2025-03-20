import torch
import matplotlib.pyplot as plt


def show_saliency(model, image):

    image.requires_grad = True

    output = model(image)

    model.zero_grad()  
    output.backward()

    saliency, _ = torch.max(image.grad.data.abs(), dim=1)

    saliency = saliency - saliency.min()
    saliency = saliency / saliency.max()
    
    image = image.detach()
    
    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image.cpu().squeeze().permute(1, 2, 0))
    ax[0].set_title("Input Image")

    ax[1].imshow(image.cpu().squeeze().permute(1, 2, 0))
    ax[1].imshow(saliency.squeeze().cpu(), cmap='jet', alpha=0.5)

    plt.show()