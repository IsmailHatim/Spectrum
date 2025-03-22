import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetClassifier(nn.Module):
    def __init__(self, model_name="densenet121", pretrained=True, device=torch.device("cpu")):
        super(DenseNetClassifier, self).__init__()
        supported_models = {
            "densenet121": models.densenet121,
            "densenet161": models.densenet161,
            "densenet169": models.densenet169,
            "densenet201": models.densenet201,
        }

        if model_name not in supported_models:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(supported_models.keys())}")
        
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        if model_name == "densenet169":
            weights = models.DenseNet169_Weights.DEFAULT if pretrained else None
        elif model_name == "densenet161":
            weights = models.DenseNet161_Weights.DEFAULT if pretrained else None
        elif model_name == "densenet201":
            weights = models.DenseNet201_Weights.DEFAULT if pretrained else None

        self.model = supported_models[model_name](weights=weights).to(device)

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