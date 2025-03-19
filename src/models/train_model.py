import torch
from tqdm import tqdm

def train_model(model, train_loader, num_epochs, criterion, optimizer, device):
    train_accuracies = []
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        train_accuracies.append(train_accuracy)
        train_losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")
    
    return train_accuracies, train_losses