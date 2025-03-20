import torch

@torch.no_grad()
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    for images, labels, _ in test_loader:
        images, labels = images.to(device), labels.to(device, dtype=torch.float32)
        outputs = model(images).squeeze()
        
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    test_accuracy = 100 * correct / total

    return test_accuracy