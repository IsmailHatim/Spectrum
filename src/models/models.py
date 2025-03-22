import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetClassifier(nn.Module):
    def __init__(self, pretrained=True, device=torch.device("cpu")):
        super(DenseNetClassifier, self).__init__()
        
        weights = "DEFAULT" if pretrained else None
        self.model = models.densenet121(weights=weights).to(device)

        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        ).to(device)
        
        self.device = device
    
    def forward(self, x):
        return self.model(x)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

class ResNetClassifier(nn.Module):
    def __init__(self, pretrained=True, device=torch.device("cpu")):
        super(ResNetClassifier, self).__init__()
        
        weights = "DEFAULT" if pretrained else None
        self.model = models.resnet50(weights=weights).to(device)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        ).to(device)
        
        self.device = device
    
    def forward(self, x):
        return self.model(x)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
