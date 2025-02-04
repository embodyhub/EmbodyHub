"""Base Actuator Module

This module defines the base Actuator class and related interfaces for the EmbodyHub framework.
All specific actuator implementations should inherit from these base classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class Actuator(ABC):
    """Base class for all actuators in the EmbodyHub framework.
    
    This class defines the core interface that all actuators must implement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the actuator.
        
        Args:
            config: Optional configuration dictionary for the actuator.
        """
        self.config = config or {}
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize actuator-specific components.
        
        This method should be implemented by subclasses to perform any
        necessary initialization.
        """
        pass
    
    @abstractmethod
    def execute(self, command: Any) -> bool:
        """Execute a command on the actuator.
        
        Args:
            command: The command to execute.
            
        Returns:
            True if the command was executed successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the actuator.
        
        Returns:
            A dictionary containing the current state information.
        """
        pass
    
    def close(self) -> None:
        """Clean up the actuator's resources."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"