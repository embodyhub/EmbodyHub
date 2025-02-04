"""Unit tests for the Agent class.

This module contains test cases for verifying the functionality of the Agent class
and its interaction with adapters and environments.
"""

import pytest
from typing import Dict, Any
from src.core.agent import Agent

def test_agent_initialization(mock_agent, mock_adapter, mock_environment):
    """Test that agent initializes correctly with adapter and environment."""
    assert mock_agent.adapter == mock_adapter
    assert mock_agent.environment == mock_environment

def test_agent_step(mock_agent):
    """Test agent's step function with mock environment."""
    observation = mock_agent.environment.reset()
    action = mock_agent.step(observation)
    assert isinstance(action, dict)

def test_agent_reset(mock_agent):
    """Test agent's reset functionality."""
    initial_state = mock_agent.reset()
    assert isinstance(initial_state, dict)
    assert initial_state["state"] == "initial"

def test_agent_training_mode(mock_agent):
    """Test agent's training mode toggle."""
    mock_agent.train()
    assert mock_agent.training
    mock_agent.eval()
    assert not mock_agent.training

def test_agent_save_load(mock_agent, tmp_path):
    """Test agent's state saving and loading functionality."""
    save_path = tmp_path / "agent_state.pt"
    mock_agent.save(save_path)
    assert save_path.exists()
    
    # Test loading state
    new_agent = Agent(adapter=mock_agent.adapter, environment=mock_agent.environment)
    new_agent.load(save_path)
    
    # Verify loaded state matches original
    assert new_agent.training == mock_agent.training