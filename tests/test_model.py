"""Tests for model architecture."""
import pytest
import torch
import torch.nn as nn
from src.model import ColonCancerClassifier, create_model


def test_model_creation():
    """Test that model can be created with default parameters."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet18',
        pretrained=False,
        dropout=0.3
    )
    
    assert model is not None
    assert isinstance(model, nn.Module)


def test_model_architecture_resnet18():
    """Test ResNet-18 architecture."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet18',
        pretrained=False
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0
    
    # ResNet-18 should have ~11M parameters
    assert 10_000_000 < total_params < 15_000_000


def test_model_architecture_resnet34():
    """Test ResNet-34 architecture."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet34',
        pretrained=False
    )
    
    assert model is not None
    total_params = sum(p.numel() for p in model.parameters())
    
    # ResNet-34 should have ~21M parameters
    assert 20_000_000 < total_params < 25_000_000


def test_model_architecture_resnet50():
    """Test ResNet-50 architecture."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet50',
        pretrained=False
    )
    
    assert model is not None
    total_params = sum(p.numel() for p in model.parameters())
    
    # ResNet-50 should have ~23M parameters
    assert 22_000_000 < total_params < 28_000_000


def test_invalid_architecture():
    """Test that invalid architecture raises ValueError."""
    with pytest.raises(ValueError):
        model = ColonCancerClassifier(
            num_classes=2,
            architecture='invalid_arch',
            pretrained=False
        )


def test_forward_pass():
    """Test forward pass with dummy input."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet18',
        pretrained=False
    )
    model.eval()
    
    # Create dummy input (batch_size=4, channels=3, height=224, width=224)
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # Check output shape
    assert output.shape == (4, 2)  # batch_size=4, num_classes=2


def test_output_range():
    """Test that output logits are reasonable."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet18',
        pretrained=False
    )
    model.eval()
    
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    # Logits should be finite
    assert torch.all(torch.isfinite(output))
    
    # After softmax, should sum to 1
    probs = torch.softmax(output, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)


def test_model_gradient_flow():
    """Test that gradients flow through the model."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet18',
        pretrained=False
    )
    model.train()
    
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_target = torch.tensor([0, 1])
    
    # Forward pass
    output = model(dummy_input)
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, dummy_target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.any(param.grad != 0), f"Zero gradient for {name}"


def test_model_dropout():
    """Test that dropout is applied during training."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet18',
        pretrained=False,
        dropout=0.5
    )
    
    # Check that dropout layers exist
    dropout_layers = [
        module for module in model.modules() 
        if isinstance(module, nn.Dropout)
    ]
    assert len(dropout_layers) > 0, "No dropout layers found"


def test_model_batch_independence():
    """Test that batch samples are processed independently."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet18',
        pretrained=False
    )
    model.eval()
    
    # Create two identical inputs
    input1 = torch.randn(1, 3, 224, 224)
    input2 = input1.clone()
    
    # Process separately
    with torch.no_grad():
        output1 = model(input1)
        output2 = model(input2)
    
    # Outputs should be identical
    assert torch.allclose(output1, output2, atol=1e-6)
    
    # Process as batch
    batch_input = torch.cat([input1, input2], dim=0)
    with torch.no_grad():
        batch_output = model(batch_input)
    
    # Batch output should match individual outputs
    assert torch.allclose(batch_output[0], output1[0], atol=1e-6)
    assert torch.allclose(batch_output[1], output2[0], atol=1e-6)


def test_model_device_transfer():
    """Test that model can be transferred to different devices."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet18',
        pretrained=False
    )
    
    # Test CPU
    model = model.to('cpu')
    assert next(model.parameters()).device.type == 'cpu'
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        assert next(model.parameters()).device.type == 'cuda'
        
        # Test forward pass on GPU
        dummy_input = torch.randn(2, 3, 224, 224).cuda()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.device.type == 'cuda'


def test_create_model_from_config():
    """Test model creation from configuration dictionary."""
    config = {
        'data': {'num_classes': 2},
        'model': {
            'architecture': 'resnet18',
            'pretrained': False,
            'dropout': 0.3
        }
    }
    
    model = create_model(config)
    
    assert model is not None
    assert isinstance(model, ColonCancerClassifier)


def test_model_trainable_parameters():
    """Test that model has trainable parameters."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet18',
        pretrained=False
    )
    
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    
    # All parameters should be trainable (pretrained=False)
    assert trainable_params == total_params
    assert trainable_params > 0


def test_model_output_classes():
    """Test that model outputs correct number of classes."""
    for num_classes in [2, 5, 9]:
        model = ColonCancerClassifier(
            num_classes=num_classes,
            architecture='resnet18',
            pretrained=False
        )
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape[1] == num_classes


def test_model_inference_mode():
    """Test model behavior in eval mode."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet18',
        pretrained=False,
        dropout=0.5
    )
    
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # Train mode - outputs may vary due to dropout
    model.train()
    output1_train = model(dummy_input)
    output2_train = model(dummy_input)
    
    # Eval mode - outputs should be deterministic
    model.eval()
    with torch.no_grad():
        output1_eval = model(dummy_input)
        output2_eval = model(dummy_input)
    
    # Eval mode outputs should be identical
    assert torch.allclose(output1_eval, output2_eval, atol=1e-6)


def test_model_weight_initialization():
    """Test that model weights are properly initialized."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet18',
        pretrained=False
    )
    
    # Check that weights are not all zeros or ones
    for name, param in model.named_parameters():
        if 'weight' in name:
            assert not torch.all(param == 0), f"{name} is all zeros"
            assert not torch.all(param == 1), f"{name} is all ones"
            assert torch.all(torch.isfinite(param)), f"{name} has inf/nan"


def test_model_classifier_head():
    """Test that custom classifier head is properly attached."""
    model = ColonCancerClassifier(
        num_classes=2,
        architecture='resnet18',
        pretrained=False,
        dropout=0.3
    )
    
    # Check that the final layer outputs 2 classes
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (1, 2)
    
    # Check that classifier has dropout and linear layers
    classifier = model.backbone.fc
    assert isinstance(classifier, nn.Sequential)
    
    has_dropout = any(isinstance(m, nn.Dropout) for m in classifier)
    has_linear = any(isinstance(m, nn.Linear) for m in classifier)
    has_relu = any(isinstance(m, nn.ReLU) for m in classifier)
    
    assert has_dropout, "Classifier missing dropout"
    assert has_linear, "Classifier missing linear layers"
    assert has_relu, "Classifier missing ReLU activation"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
