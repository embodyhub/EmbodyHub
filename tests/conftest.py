"""Configuration file for pytest.

This module contains fixtures and configurations that are shared across test files.
"""

import pytest
from typing import Dict, Any
from src.core.adapter import Adapter
from src.core.agent import Agent
from src.core.environment import Environment

@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Fixture that provides a basic configuration for testing."""
    return {
        "name": "test_agent",
        "model_path": "test_models",
        "max_steps": 100
    }

@pytest.fixture
def mock_adapter(mock_config) -> Adapter:
    """Fixture that provides a mock adapter for testing."""
    class MockAdapter(Adapter):
        def _initialize(self) -> None:
            pass
            
        def convert_observation(self, observation: Any) -> Any:
            return observation
            
        def convert_action(self, action: Any) -> Any:
            return action
    
    return MockAdapter(mock_config)

@pytest.fixture
def mock_environment() -> Environment:
    """Fixture that provides a mock environment for testing."""
    class MockEnvironment(Environment):
        def reset(self):
            return {"state": "initial"}
            
        def step(self, action):
            return {"state": "next"}, 0.0, False, {}
            
        def render(self):
            pass
            
        def close(self):
            pass
    
    return MockEnvironment()

@pytest.fixture
def mock_agent(mock_adapter, mock_environment) -> Agent:
    """Fixture that provides a mock agent for testing."""
    return Agent(adapter=mock_adapter, environment=mock_environment)