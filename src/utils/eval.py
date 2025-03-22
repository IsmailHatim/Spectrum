import numpy as np
from sklearn.metrics import roc_auc_score

def compute_iou_score(heatmap, ground_truth):
    
    ground_truth = np.array(ground_truth[0])
    heatmap = np.array(heatmap)
    
    intersection = np.logical_and(heatmap, ground_truth).sum() # pixels activated both in the heatmap AND the ground truth
    union = np.logical_or(heatmap, ground_truth).sum() # pixels activated wether in the heatmap OR the ground truth

    jaccard = round(intersection / union if union != 0 else 0, 2)

    return jaccard

def compute_f1_score(heatmap, ground_truth):

    return round((2*compute_iou_score(heatmap, ground_truth)) / (1 + compute_iou_score(heatmap, ground_truth)), 2)

def compute_auc_score(heatmap, ground_truth):

    heatmap = np.array(heatmap)
    ground_truth = np.array(ground_truth[0, 0])

    heatmap_flat = heatmap.flatten()
    ground_truth_flatten = ground_truth.flatten()

    auc_score = roc_auc_score(heatmap_flat, ground_truth_flatten)
    
    return auc_score