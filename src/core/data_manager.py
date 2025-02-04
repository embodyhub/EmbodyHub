"""Data Flow Management Module

This module provides the core functionality for managing multimodal data flow
in embodied AI applications.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from queue import Queue
import threading
from .data_persistence import DataPersistence

class DataManager(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._data_queues: Dict[str, Queue] = {}
        self._subscribers: Dict[str, List[callable]] = {}
        self._running = False
        self._persistence = DataPersistence(config)
        self._initialize()
    
    def publish(self, stream_name: str, data: Any) -> None:
        if stream_name not in self._data_queues:
            raise KeyError(f"Stream '{stream_name}' does not exist")
        self._data_queues[stream_name].put(data)
        self._persistence.save_data(stream_name, data)
        self._notify_subscribers(stream_name, data)
    
    def load_historical_data(self, stream_name: str, **kwargs) -> list:
        """Load historical data for a specific stream.
        
        Args:
            stream_name: Name of the stream to load data from.
            **kwargs: Additional arguments to pass to the persistence layer.
            
        Returns:
            List of historical data entries.
        """
        return self._persistence.load_data(stream_name, **kwargs)
    
    def backup_data(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of all stream data.
        
        Args:
            backup_path: Optional path for the backup file.
            
        Returns:
            Path to the created backup file.
        """
        return self._persistence.backup(backup_path)
    
    def restore_data(self, backup_path: str) -> None:
        """Restore data from a backup file.
        
        Args:
            backup_path: Path to the backup file.
        """
        self._persistence.restore(backup_path)
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize data manager specific components.
        
        This method should be implemented by subclasses to perform any
        necessary initialization.
        """
        pass
    
    def create_stream(self, stream_name: str, maxsize: int = 100) -> None:
        """Create a new data stream.
        
        Args:
            stream_name: The name of the data stream.
            maxsize: Maximum size of the stream buffer.
        """
        if stream_name in self._data_queues:
            raise ValueError(f"Stream '{stream_name}' already exists")
        self._data_queues[stream_name] = Queue(maxsize=maxsize)
        self._subscribers[stream_name] = []
    
    def subscribe(self, stream_name: str, callback: callable) -> None:
        """Subscribe to a data stream.
        
        Args:
            stream_name: The name of the stream to subscribe to.
            callback: Function to be called when new data arrives.
            
        Raises:
            KeyError: If the specified stream does not exist.
        """
        if stream_name not in self._subscribers:
            raise KeyError(f"Stream '{stream_name}' does not exist")
        self._subscribers[stream_name].append(callback)
    
    def _notify_subscribers(self, stream_name: str, data: Any) -> None:
        """Notify all subscribers of new data.
        
        Args:
            stream_name: The name of the stream with new data.
            data: The new data.
        """
        for callback in self._subscribers[stream_name]:
            try:
                callback(data)
            except Exception as e:
                print(f"Error in subscriber callback: {e}")
    
    def start(self) -> None:
        """Start the data manager."""
        self._running = True
        self._start_processing()
    
    def stop(self) -> None:
        """Stop the data manager."""
        self._running = False
    
    @abstractmethod
    def _start_processing(self) -> None:
        """Start processing data streams.
        
        This method should be implemented by subclasses to handle
        the actual data processing logic.
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(streams={list(self._data_queues.keys())}, config={self.config})"