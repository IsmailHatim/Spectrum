import os
import torch 
from torch.utils.data import Dataset
from PIL import Image

class DAGMDataset(Dataset):
    def __init__(self, root_dir, split="Train", transform=None):
        """
        Args:
            root_dir (str): Chemin du dossier dataset.
            split (str): "Train" ou "Test".
            transform (callable, optional): Transformation appliquée aux images.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = []
        
        img_dir = os.path.join(root_dir, split)
        label_dir = os.path.join(img_dir, "Label")
        
        images = [f for f in os.listdir(img_dir) if f.endswith((".PNG"))]
        
        for img_name in images:
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, img_name.replace(".PNG", "_label.PNG"))
            
            if os.path.exists(label_path):
                label = 1 
            else:
                label = 0
                label_path = os.path.join(label_dir, "dummy.png")
            
            self.data.append((img_path, label, label_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, label_path = self.data[idx]
        
        img = Image.open(img_path).convert("RGB")
        label_img = Image.open(label_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
            label_img = self.transform(label_img)

        return img, label, label_img