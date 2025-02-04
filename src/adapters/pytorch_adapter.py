"""PyTorch Adapter Module

This module provides a concrete adapter implementation for integrating PyTorch models
into the EmbodyHub framework.
"""

from typing import Any, Dict, Optional
try:
    import torch
except ImportError:
    raise ImportError("Failed to import PyTorch. Please ensure PyTorch is properly installed. You can install it using 'pip install torch'."
import numpy as np

from ..core.adapter import Adapter

class PyTorchAdapter(Adapter):
    """Adapter for integrating PyTorch models with EmbodyHub.
    
    This adapter handles the conversion between PyTorch tensors and EmbodyHub's
    standard format, as well as managing PyTorch model instances.
    """
    
    def _initialize(self) -> None:
        """Initialize PyTorch-specific components.
        
        Sets up device configuration and other PyTorch-specific settings.
        """
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    def register_model(self, name: str, model: torch.nn.Module) -> None:
        """Register a PyTorch model with the adapter.
        
        Args:
            name: The name to register the model under.
            model: The PyTorch model instance to register.
            
        Raises:
            TypeError: If the provided model is not a PyTorch Module.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model must be a PyTorch Module instance")
        model.to(self.device)
        super().register_model(name, model)
    
    def convert_observation(self, observation: Any) -> torch.Tensor:
        """Convert an observation to PyTorch tensor format.
        
        Args:
            observation: The observation in EmbodyHub format (typically numpy array).
            
        Returns:
            The observation converted to a PyTorch tensor.
        """
        if isinstance(observation, torch.Tensor):
            return observation.to(self.device)
        elif isinstance(observation, np.ndarray):
            return torch.from_numpy(observation).to(self.device)
        else:
            return torch.tensor(observation, device=self.device)
    
    def convert_action(self, action: torch.Tensor) -> np.ndarray:
        """Convert a PyTorch tensor action to numpy array.
        
        Args:
            action: The action as a PyTorch tensor.
            
        Returns:
            The action converted to a numpy array.
        """
        if isinstance(action, torch.Tensor):
            return action.detach().cpu().numpy()
        return action
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, config={self.config})"