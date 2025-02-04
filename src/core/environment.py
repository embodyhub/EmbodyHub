"""Base Environment Module

This module defines the base Environment class and related interfaces for the EmbodyHub framework.
All specific environment implementations should inherit from these base classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

class Environment(ABC):
    """Base class for all environments in the EmbodyHub framework.
    
    This class defines the core interface that all environments must implement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the environment.
        
        Args:
            config: Optional configuration dictionary for the environment.
        """
        self.config = config or {}
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize environment-specific components.
        
        This method should be implemented by subclasses to perform any
        necessary initialization.
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Execute one time step within the environment.
        
        Args:
            action: An action provided by the agent.
            
        Returns:
            A tuple containing:
            - observation: The current observation of the environment
            - reward: Amount of reward achieved by the previous action
            - done: Whether the episode has ended
            - info: Contains auxiliary diagnostic information
        """
        pass
    
    @abstractmethod
    def reset(self) -> Any:
        """Reset the environment to an initial state.
        
        Returns:
            observation: The initial observation.
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = 'human') -> Optional[Any]:
        """Render the environment.
        
        Args:
            mode: The mode to render with (e.g. 'human', 'rgb_array', 'ansi').
            
        Returns:
            Optional rendered output depending on the mode.
        """
        pass
    
    def close(self) -> None:
        """Clean up the environment's resources."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"