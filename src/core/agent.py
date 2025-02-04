"""Base Agent Module

This module defines the base Agent class and related interfaces for the EmbodyHub framework.
All specific agent implementations should inherit from these base classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class Agent(ABC):
    """Base class for all agents in the EmbodyHub framework.
    
    This class defines the core interface that all agents must implement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent.
        
        Args:
            config: Optional configuration dictionary for the agent.
        """
        self.config = config or {}
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize agent-specific components.
        
        This method should be implemented by subclasses to perform any
        necessary initialization.
        """
        pass
    
    @abstractmethod
    def step(self, observation: Any) -> Any:
        """Process a single step with the given observation.
        
        Args:
            observation: The current observation from the environment.
            
        Returns:
            The agent's action or response to the observation.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the agent to its initial state.
        
        This method should be called when starting a new episode or session.
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"