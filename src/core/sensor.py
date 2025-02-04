"""Base Sensor Module

This module defines the base Sensor class and related interfaces for the EmbodyHub framework.
All specific sensor implementations should inherit from these base classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class Sensor(ABC):
    """Base class for all sensors in the EmbodyHub framework.
    
    This class defines the core interface that all sensors must implement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the sensor.
        
        Args:
            config: Optional configuration dictionary for the sensor.
        """
        self.config = config or {}
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize sensor-specific components.
        
        This method should be implemented by subclasses to perform any
        necessary initialization.
        """
        pass
    
    @abstractmethod
    def read(self) -> Any:
        """Read current sensor data.
        
        Returns:
            The current sensor reading in the appropriate format.
        """
        pass
    
    @abstractmethod
    def calibrate(self) -> bool:
        """Calibrate the sensor.
        
        Returns:
            True if calibration was successful, False otherwise.
        """
        pass
    
    def close(self) -> None:
        """Clean up the sensor's resources."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"