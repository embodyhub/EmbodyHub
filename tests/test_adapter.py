"""Unit tests for the Adapter class.

This module contains test cases for verifying the functionality of the base Adapter class
and its core features.
"""

# Import pytest testing framework
try:
    import pytest
except ImportError:
    raise ImportError("Please install pytest first: pip install pytest")
from typing import Dict, Any

def test_adapter_initialization(mock_adapter):
    """Test that adapter initializes correctly with config."""
    assert mock_adapter.config["name"] == "test_agent"
    assert mock_adapter.config["model_path"] == "test_models"
    assert mock_adapter._models == {}

def test_model_registration(mock_adapter):
    """Test model registration and retrieval functionality."""
    test_model = {"type": "test_model"}
    mock_adapter.register_model("test", test_model)
    
    # Test successful retrieval
    assert mock_adapter.get_model("test") == test_model
    
    # Test retrieval of non-existent model
    with pytest.raises(KeyError):
        mock_adapter.get_model("non_existent")

def test_observation_conversion(mock_adapter):
    """Test observation conversion functionality."""
    test_observation = {"sensor_data": [1, 2, 3]}
    converted = mock_adapter.convert_observation(test_observation)
    assert converted == test_observation

def test_action_conversion(mock_adapter):
    """Test action conversion functionality."""
    test_action = {"command": "move", "params": {"direction": "forward"}}
    converted = mock_adapter.convert_action(test_action)
    assert converted == test_action

def test_adapter_representation(mock_adapter):
    """Test string representation of adapter."""
    expected_repr = f"MockAdapter(config={mock_adapter.config})"
    assert repr(mock_adapter) == expected_repr