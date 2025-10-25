"""Tests for utility functions."""
import pytest
import torch
import json
from pathlib import Path
from src.utils import set_seed, AverageMeter, save_json, load_json


def test_set_seed():
    """Test seed setting for reproducibility."""
    set_seed(42)
    x1 = torch.rand(10)
    
    set_seed(42)
    x2 = torch.rand(10)
    
    assert torch.allclose(x1, x2), "Seed not working correctly"


def test_set_seed_different():
    """Test that different seeds produce different results."""
    set_seed(42)
    x1 = torch.rand(10)
    
    set_seed(123)
    x2 = torch.rand(10)
    
    assert not torch.allclose(x1, x2), "Different seeds should produce different results"


def test_average_meter():
    """Test AverageMeter class."""
    meter = AverageMeter()
    
    # Initial state
    assert meter.avg == 0
    assert meter.count == 0
    
    # Single update
    meter.update(10, n=1)
    assert meter.avg == 10
    assert meter.val == 10
    
    # Multiple updates
    meter.update(20, n=1)
    assert meter.avg == 15
    assert meter.count == 2
    
    # Reset
    meter.reset()
    assert meter.avg == 0
    assert meter.count == 0


def test_average_meter_weighted():
    """Test AverageMeter with weighted updates."""
    meter = AverageMeter()
    
    meter.update(10, n=2)  # 10 * 2 = 20
    meter.update(20, n=3)  # 20 * 3 = 60
    # Average = (20 + 60) / (2 + 3) = 80 / 5 = 16
    
    assert meter.avg == 16
    assert meter.count == 5


def test_save_and_load_json(tmp_path):
    """Test JSON save and load functions."""
    test_data = {
        'accuracy': 0.95,
        'loss': 0.15,
        'epochs': 30,
        'config': {'batch_size': 32}
    }
    
    filepath = tmp_path / 'test.json'
    
    # Save
    save_json(test_data, str(filepath))
    assert filepath.exists()
    
    # Load
    loaded_data = load_json(str(filepath))
    assert loaded_data == test_data


def test_save_json_creates_directory(tmp_path):
    """Test that save_json creates parent directories."""
    filepath = tmp_path / 'subdir' / 'nested' / 'test.json'
    test_data = {'key': 'value'}
    
    save_json(test_data, str(filepath))
    
    assert filepath.exists()
    assert filepath.parent.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
