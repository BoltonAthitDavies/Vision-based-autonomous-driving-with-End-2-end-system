"""
Unit tests for the PilotNet model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import pytest
from src.models.pilotnet import PilotNet, SimplePilotNet


def test_pilotnet_creation():
    """Test PilotNet model creation"""
    model = PilotNet(input_channels=3, output_size=1, dropout_rate=0.5)
    assert model is not None
    assert isinstance(model, PilotNet)


def test_pilotnet_forward_pass():
    """Test PilotNet forward pass with correct input shape"""
    model = PilotNet(input_channels=3, output_size=1)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 66, 200)
    
    output = model(dummy_input)
    
    assert output.shape == (batch_size, 1)


def test_pilotnet_multiple_outputs():
    """Test PilotNet with multiple outputs"""
    model = PilotNet(input_channels=3, output_size=3)
    dummy_input = torch.randn(2, 3, 66, 200)
    
    output = model(dummy_input)
    
    assert output.shape == (2, 3)


def test_simple_pilotnet_creation():
    """Test SimplePilotNet model creation"""
    model = SimplePilotNet(input_channels=3, output_size=1)
    assert model is not None
    assert isinstance(model, SimplePilotNet)


def test_simple_pilotnet_forward_pass():
    """Test SimplePilotNet forward pass"""
    model = SimplePilotNet(input_channels=3, output_size=1)
    dummy_input = torch.randn(2, 3, 66, 200)
    
    output = model(dummy_input)
    
    assert output.shape == (2, 1)


def test_model_parameter_count():
    """Test that model has trainable parameters"""
    model = PilotNet(input_channels=3, output_size=1)
    param_count = model.count_parameters()
    
    assert param_count > 0


def test_model_save_load():
    """Test model save and load functionality"""
    model = PilotNet(input_channels=3, output_size=1)
    
    # Save model
    save_path = '/tmp/test_model.pth'
    model.save(save_path)
    
    # Load model
    new_model = PilotNet(input_channels=3, output_size=1)
    new_model.load(save_path)
    
    # Verify weights are the same
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)
    
    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)


if __name__ == '__main__':
    pytest.main([__file__])
