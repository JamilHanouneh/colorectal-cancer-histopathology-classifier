"""Tests for preprocessing and data loading functions."""
import pytest
import torch
from PIL import Image
import numpy as np
from src.data_loader import get_transforms, ColonCancerDataset


def test_train_transforms():
    """Test training transformations."""
    config = {
        'data': {'image_size': [224, 224]},
        'augmentation': {
            'horizontal_flip': True,
            'vertical_flip': True,
            'rotation_degrees': 20,
            'brightness': 0.2,
            'contrast': 0.2
        }
    }
    
    train_transform = get_transforms(config, is_train=True)
    
    assert train_transform is not None
    assert callable(train_transform)


def test_eval_transforms():
    """Test evaluation transformations."""
    config = {
        'data': {'image_size': [224, 224]},
        'augmentation': {
            'horizontal_flip': True,
            'vertical_flip': True,
            'rotation_degrees': 20,
            'brightness': 0.2,
            'contrast': 0.2
        }
    }
    
    eval_transform = get_transforms(config, is_train=False)
    
    assert eval_transform is not None
    assert callable(eval_transform)


def test_transform_output_shape():
    """Test that transforms produce correct tensor shape."""
    config = {
        'data': {'image_size': [224, 224]},
        'augmentation': {
            'horizontal_flip': False,
            'vertical_flip': False,
            'rotation_degrees': 0,
            'brightness': 0,
            'contrast': 0
        }
    }
    
    transform = get_transforms(config, is_train=False)
    
    # Create dummy image
    dummy_image = Image.new('RGB', (512, 512), color='red')
    
    # Apply transform
    transformed = transform(dummy_image)
    
    # Check output
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 224, 224)  # C, H, W
    assert transformed.dtype == torch.float32


def test_transform_normalization():
    """Test that transforms normalize values correctly."""
    config = {
        'data': {'image_size': [224, 224]},
        'augmentation': {
            'horizontal_flip': False,
            'vertical_flip': False,
            'rotation_degrees': 0,
            'brightness': 0,
            'contrast': 0
        }
    }
    
    transform = get_transforms(config, is_train=False)
    
    # Create white image
    white_image = Image.new('RGB', (224, 224), color=(255, 255, 255))
    
    transformed = transform(white_image)
    
    # After normalization, values should not be in [0, 1] range
    assert transformed.min() < 0 or transformed.max() > 1


def test_dataset_creation():
    """Test ColonCancerDataset creation."""
    image_paths = ['path1.tif', 'path2.tif', 'path3.tif']
    labels = [0, 1, 0]
    
    dataset = ColonCancerDataset(image_paths, labels, transform=None)
    
    assert len(dataset) == 3
    assert dataset.image_paths == image_paths
    assert dataset.labels == labels


def test_dataset_length():
    """Test dataset length property."""
    image_paths = ['img1.tif'] * 10
    labels = [0] * 10
    
    dataset = ColonCancerDataset(image_paths, labels)
    
    assert len(dataset) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
