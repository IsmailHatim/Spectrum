
import matplotlib.pyplot as plt
import sys
import torch
import importlib
import numpy as np
import time
import torchvision.transforms as transforms
from skimage.segmentation import mark_boundaries
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from src.data.dataset import DAGMDataset
from src.models.models import DenseNetClassifier
from src.utils.eval import compute_iou_score, compute_f1_score, compute_auc_score
from tqdm import tqdm


sys.dont_write_bytecode = True

def main(args):
    CLASS = 1
    IMAGE_PATH = f"data/dataset/Class{CLASS}/"
    MODEL_PATH = f"data/models/model_{args.model_name}_class{CLASS}.pth"

    method = getattr(importlib.import_module(f"src.task.{args.method}"), f"show_{args.method}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    test_dataset = DAGMDataset(root_dir=IMAGE_PATH, split="Test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNetClassifier(model_name=args.model_name, pretrained=True, device=device)
    model.load_model(MODEL_PATH)

    iou_scores = []
    f1_scores = []
    auc_scores = []

    for batch in tqdm(test_loader):

        image, label, label_image = batch
        if label ==1:
            start_time = time.perf_counter()
            explanation, explanation_thresholded = method(model.model, image, device, threshold=args.threshold, conv_layer_index=args.conv_layer_index)
            end_time = time.perf_counter()

            execution_time = end_time - start_time

            print("+" + "-" * 50 + "+")
            print(f'True Label: {label}')
            if args.method != "lime":
                iou_score = compute_iou_score(explanation_thresholded, label_image)
                f1_score = compute_f1_score(explanation_thresholded, label_image)
                # auc_score = compute_auc_score(explanation, label_image)
                iou_scores.append(iou_score)
                f1_scores.append(f1_score)
                # auc_scores.append(auc_score)

                if args.method == 'saliency':
                    image = image.detach()
                
                
                print(f"IoU Score : {iou_score}")
                print(f"F1 Score : {f1_score}")
                # print(f"ROC AUC Score : {auc_score}")
            print(f"Execution Time : {execution_time:.4f} seconds")
            print("+" + "-" * 50 + "+")

    print(f"Average IoU Score : {sum(iou_scores)/len(iou_scores)}")
    print(f"Standard Deviation IoU Score : {np.std(iou_scores)}")
    print(f"Average F1 Score : {sum(f1_scores)/len(f1_scores)}")
    print(f"Standard Deviation F1 Score : {np.std(f1_scores)}")
    # print(f"Average ROC AUC Score : {sum(auc_scores)/len(auc_scores)}")
    # print(f"Standard Deviation ROC AUC Score : {np.std(auc_scores)}")

if __name__ == "__main__":
    parser = ArgumentParser(description='Run selected explanation method on DAGM Dataset.')

    parser.add_argument('--method', default='gradcam', type=str, help='Explanation method to use')
    parser.add_argument('--threshold', default='0.5', type=float, help='Threshold used to compute binary mask')
    parser.add_argument('--model_name', default='densenet121', type=str, help='Model name to use')
    parser.add_argument('--conv_layer_index', default=-2, type=int, help='Index of the convolutional layer to analyze')

    args = parser.parse_args()
    
    main(args)


