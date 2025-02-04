"""Unit tests for the Environment class.

This module contains test cases for verifying the functionality of the Environment class
and its core features for managing embodied AI environments.
"""

try:
    import pytest
except ImportError:
    # If pytest is not installed, prompt for installation
    raise ImportError("Please install pytest first: pip install pytest")
from typing import Dict, Any
from src.core.environment import Environment

def test_environment_initialization(mock_environment):
    """Test that environment initializes correctly."""
    assert isinstance(mock_environment, Environment)

def test_environment_reset(mock_environment):
    """Test environment reset functionality."""
    initial_state = mock_environment.reset()
    assert isinstance(initial_state, dict)
    assert initial_state["state"] == "initial"

def test_environment_step(mock_environment):
    """Test environment step functionality."""
    action = {"command": "test_action"}
    observation, reward, done, info = mock_environment.step(action)
    
    assert isinstance(observation, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    assert observation["state"] == "next"
    assert reward == 0.0
    assert not done

def test_environment_render(mock_environment):
    """Test environment rendering functionality."""
    # Should not raise any exceptions
    mock_environment.render()

def test_environment_close(mock_environment):
    """Test environment cleanup functionality."""
    # Should not raise any exceptions
    mock_environment.close()