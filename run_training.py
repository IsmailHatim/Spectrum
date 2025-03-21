import sys
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import time

from src.data.dataset import DAGMDataset
from src.models.models import DenseNetClassifier
from src.models.train_model import train_model
from src.models.evaluation import evaluate_model


def main(args):
    print(f"Training model: {args.model_name} on Class {args.img_class}")
    print("+" + "-" * 50 + "+")
    
    # Setup paths
    IMAGE_PATH = f"data/dataset/Class{args.img_class}/"
    MODEL_PATH = f"data/models/model_{args.model_name}_class{args.img_class}.pth"
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup transforms and datasets
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    train_dataset = DAGMDataset(root_dir=IMAGE_PATH, split="Train", transform=transform)
    test_dataset = DAGMDataset(root_dir=IMAGE_PATH, split="Test", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Initialize model
    if args.model_name == "densenet121":
        model = DenseNetClassifier(pretrained=True, device=device)
    else:
        raise ValueError(f"Model {args.model_name} is not supported yet.")
    
    # Setup training parameters
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    train_accuracies, train_losses = train_model(
        model.model, 
        train_loader, 
        args.epochs, 
        criterion, 
        optimizer, 
        device
    )
    training_time = time.time() - start_time
    
    # Save the model
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Evaluate on test data
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training metrics
    if args.plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig(f"data/figures/training_metrics_{args.model_name}_class{args.img_class}.png")
        plt.show()
    
    print("+" + "-" * 50 + "+")


if __name__ == "__main__":
    parser = ArgumentParser(description='Train and save a model on DAGM Dataset.')
    
    parser.add_argument('--model_name', default='densenet121', type=str, help='Model architecture to use')
    parser.add_argument('--img_class', default=1, type=int, help='Image class from DAGM Dataset')
    parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate for optimizer')
    parser.add_argument('--plot', action='store_true', help='Plot and save training metrics')
    
    args = parser.parse_args()
    
    main(args)
