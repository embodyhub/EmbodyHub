"""EmbodyHub Adapter Module

This module provides the base adapter class for integrating different embodied AI frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

class Adapter(ABC):
    """Base class for all adapters in the EmbodyHub framework.
    
    This class defines the standard interface for integrating different embodied AI frameworks.
    Adapters are responsible for converting between EmbodyHub's standard format and
    framework-specific formats.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the adapter.
        
        Args:
            config: Optional configuration dictionary for the adapter.
        """
        self.config = config or {}
        self._models = {}
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize adapter-specific components.
        
        This method should be implemented by subclasses to perform any
        necessary initialization.
        """
        pass
    
    def register_model(self, name: str, model: Any) -> None:
        """Register a model with the adapter.
        
        Args:
            name: The name to register the model under.
            model: The model instance to register.
        """
        self._models[name] = model
    
    def get_model(self, name: str) -> Any:
        """Retrieve a registered model.
        
        Args:
            name: The name of the model to retrieve.
            
        Returns:
            The registered model instance.
            
        Raises:
            KeyError: If no model is registered under the given name.
        """
        if name not in self._models:
            raise KeyError(f"No model registered under name '{name}'")
        return self._models[name]
    
    @abstractmethod
    def convert_observation(self, observation: Any) -> Any:
        """Convert an observation from framework-specific format to EmbodyHub format.
        
        Args:
            observation: The observation in framework-specific format.
            
        Returns:
            The observation converted to EmbodyHub format.
        """
        pass
    
    @abstractmethod
    def convert_action(self, action: Any) -> Any:
        """Convert an action from EmbodyHub format to framework-specific format.
        
        Args:
            action: The action in EmbodyHub format.
            
        Returns:
            The action converted to framework-specific format.
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"