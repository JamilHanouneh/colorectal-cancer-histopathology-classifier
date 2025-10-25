"""Data loading and preprocessing module for NCT-CRC-HE-100K dataset."""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
from sklearn.model_selection import train_test_split


class ColonCancerDataset(Dataset):
    """Dataset class for NCT-CRC-HE-100K histopathological images."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            transform: Optional image transformations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (image tensor, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(config: dict, is_train: bool = True) -> transforms.Compose:
    """
    Get image transformations.
    
    Args:
        config: Configuration dictionary
        is_train: Whether for training set
        
    Returns:
        Composed transformations
    """
    image_size = config['data']['image_size']
    
    if is_train and config['augmentation']['horizontal_flip']:
        transform_list = [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip() if config['augmentation']['vertical_flip'] else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(config['augmentation']['rotation_degrees']),
            transforms.ColorJitter(
                brightness=config['augmentation']['brightness'],
                contrast=config['augmentation']['contrast']
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    else:
        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    
    return transforms.Compose(transform_list)


def prepare_data(config: dict, seed: int = 42) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """
    Prepare and split NCT-CRC-HE-100K dataset.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
        
    Returns:
        Train, val, test image paths and labels
    """
    raw_dir = Path(config['data']['raw_dir'])
    
    # NCT-CRC-HE-100K dataset structure
    # After extraction: NCT-CRC-HE-100K/<class_name>/*.tif
    dataset_dir = raw_dir / 'NCT-CRC-HE-100K'
    
    # Classes: NORM (normal colon mucosa) vs TUM (colorectal adenocarcinoma)
    classes = config['data']['classes']
    
    # Collect image paths and labels
    image_paths = []
    labels = []
    
    # Class 0: Normal colon mucosa (NORM)
    norm_dir = dataset_dir / 'NORM'
    if norm_dir.exists():
        for img_path in norm_dir.glob('*.tif'):
            image_paths.append(str(img_path))
            labels.append(0)
    
    # Class 1: Colorectal adenocarcinoma epithelium (TUM)
    tum_dir = dataset_dir / 'TUM'
    if tum_dir.exists():
        for img_path in tum_dir.glob('*.tif'):
            image_paths.append(str(img_path))
            labels.append(1)
    
    print(f"Found {len([l for l in labels if l == 0])} normal images")
    print(f"Found {len([l for l in labels if l == 1])} tumor images")
    
    if len(image_paths) == 0:
        raise ValueError(
            f"No images found in {dataset_dir}. "
            f"Please download NCT-CRC-HE-100K from Zenodo and extract it to {raw_dir}"
        )
    
    # Split data
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']
    
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels,
        test_size=test_split,
        random_state=seed,
        stratify=labels
    )
    
    # Second split: separate train and validation
    val_ratio = val_split / (train_split + val_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data_loaders(config: dict, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
        
    Returns:
        Train, validation, and test data loaders
    """
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(config, seed)
    
    # Get transformations
    train_transform = get_transforms(config, is_train=True)
    eval_transform = get_transforms(config, is_train=False)
    
    # Create datasets
    train_dataset = ColonCancerDataset(X_train, y_train, train_transform)
    val_dataset = ColonCancerDataset(X_val, y_val, eval_transform)
    test_dataset = ColonCancerDataset(X_test, y_test, eval_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
