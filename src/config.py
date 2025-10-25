"""Configuration management module."""
import yaml
from pathlib import Path
from typing import Dict, Any
import torch


class Config:
    """Configuration handler for the project."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file."""
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)
        
        # Set device
        device_name = self.config['hardware']['device']
        if device_name == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device_name = 'cpu'
        self.device = torch.device(device_name)
        
    def __getitem__(self, key: str) -> Any:
        """Access config values using dictionary syntax."""
        return self.config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self.config.get(key, default)
    
    @property
    def seed(self) -> int:
        """Get random seed."""
        return self.config['project']['seed']
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.config['data']['batch_size']
    
    @property
    def num_epochs(self) -> int:
        """Get number of training epochs."""
        return self.config['training']['epochs']
    
    @property
    def learning_rate(self) -> float:
        """Get learning rate."""
        return self.config['model']['learning_rate']
