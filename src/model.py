"""Model architecture module."""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ColonCancerClassifier(nn.Module):
    """ResNet-based classifier for colon cancer detection."""
    
    def __init__(
        self,
        num_classes: int = 2,
        architecture: str = 'resnet18',
        pretrained: bool = False,
        dropout: float = 0.3
    ):
        """
        Initialize classifier.
        
        Args:
            num_classes: Number of output classes
            architecture: Model architecture name
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
        """
        super(ColonCancerClassifier, self).__init__()
        
        # Load base model
        if architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif architecture == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Replace final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        return self.backbone(x)


def create_model(config: dict) -> ColonCancerClassifier:
    """
    Create model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model = ColonCancerClassifier(
        num_classes=config['data']['num_classes'],
        architecture=config['model']['architecture'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
    
    return model
