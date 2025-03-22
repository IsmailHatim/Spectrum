
import matplotlib.pyplot as plt
import sys
import torch
import importlib
import time
import torchvision.transforms as transforms
from skimage.segmentation import mark_boundaries
from argparse import ArgumentParser
from src.data.dataset import DAGMDataset
from src.models.models import DenseNetClassifier
from src.utils.eval import compute_iou_score, compute_f1_score, compute_auc_score


sys.dont_write_bytecode = True

def main(args):
    IMAGE_PATH = f"data/dataset/Class{args.img_class}/"
    MODEL_PATH = f"data/models/model_{args.model_name}_class{args.img_class}.pth"
    INDEX = args.index

    method = getattr(importlib.import_module(f"src.task.{args.method}"), f"show_{args.method}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = DAGMDataset(root_dir=IMAGE_PATH, split="Train", transform=transform)
    test_dataset = DAGMDataset(root_dir=IMAGE_PATH, split="Test", transform=transform)

    image, label, label_image = test_dataset[INDEX][0].unsqueeze(0), test_dataset[INDEX][1], test_dataset[INDEX][2].unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = DenseNetClassifier(model_name=args.model_name, pretrained=True, device=device)
    model.load_model(MODEL_PATH)
    
    model.eval()
    output = model(image.to(device))
    prob = output.item()
    
    start_time = time.perf_counter()
    explanation, explanation_thresholded = method(model.model, image, device, threshold=args.threshold, conv_layer_index=args.conv_layer_index)
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    if args.method != "lime":
        iou_score = compute_iou_score(explanation_thresholded, label_image)
        f1_score = compute_f1_score(explanation_thresholded, label_image)
        # auc_score = compute_auc_score(explanation, label_image)
        
        if args.method == 'saliency':
            image = image.detach()
        
        print("+" + "-" * 50 + "+")
        print(f'True Label: {label}')
        print(f"IoU Score : {iou_score}")
        print(f"F1 Score : {f1_score}")
        # print(f"ROC AUC Score : {auc_score}")
        print(f"Execution Time : {execution_time:.4f} seconds")
        print("+" + "-" * 50 + "+")

        _, ax = plt.subplots(1, 4, figsize=(10, 5))
        ax[0].imshow(image.cpu().squeeze().permute(1, 2, 0))
        ax[0].set_title("Input Image")

        ax[1].imshow(image.cpu().squeeze().permute(1, 2, 0))
        ax[1].imshow(explanation, cmap='jet', alpha=0.5)
        ax[1].set_title(f"{args.method.capitalize()} Prob: {prob*100:.1f}%")
        ax[1].legend()

        ax[2].imshow(explanation_thresholded, cmap='gray')
        ax[2].set_title(f"Thresholded Mask")

        ax[3].imshow(label_image.squeeze().permute(1, 2, 0))
        ax[3].set_title(f"Ground Truth")
    else:

        print("+" + "-" * 50 + "+")
        print(f"Execution Time : {execution_time:.4f} seconds")
        print("+" + "-" * 50 + "+")
        print(f'True Label: {label}')

        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(image.cpu().squeeze().permute(1, 2, 0))
        ax[0].set_title("Input Image")
        
        ax[1].imshow(mark_boundaries(explanation, explanation_thresholded))
        ax[1].set_title(f"LIME Prob: {prob*100:.1f}%")
        
        ax[2].imshow(label_image.squeeze().permute(1, 2, 0))
        ax[2].set_title(f"Ground Truth")

    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description='Run selected explanation method on DAGM Dataset.')

    parser.add_argument('--method', default='gradcam', type=str, help='Explanation method to use')
    parser.add_argument('--index', default='0', type=int, help='Image index from DAGM Dataset')
    parser.add_argument('--threshold', default='0.5', type=float, help='Threshold used to compute binary mask')
    parser.add_argument('--img_class', default='1', type=int, help='Image class from DAGM Dataset')
    parser.add_argument('--model_name', default='densenet121', type=str, help='Model name to use')
    parser.add_argument('--conv_layer_index', default=-2, type=int, help='Index of the convolutional layer to analyze')

    args = parser.parse_args()
    
    main(args)


