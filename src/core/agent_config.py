"""Agent Configuration Module

This module provides functionality for managing agent configurations
in the EmbodyHub framework.
"""

from typing import Dict, Any, Optional
import json
import os
from threading import Lock

class AgentConfig:
    """Manages agent configurations and their dynamic updates.
    
    This class handles loading, updating, and validating agent configurations.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the configuration manager.
        
        Args:
            config_path: Optional path to the configuration directory.
        """
        self._config_path = config_path or 'configs'
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize configuration manager components."""
        os.makedirs(self._config_path, exist_ok=True)
    
    def load_config(self, agent_id: str) -> Dict[str, Any]:
        """Load configuration for a specific agent.
        
        Args:
            agent_id: ID of the agent to load configuration for.
            
        Returns:
            The loaded configuration dictionary.
        """
        with self._lock:
            config_file = os.path.join(self._config_path, f"{agent_id}.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self._configs[agent_id] = config
                return config
            return {}
    
    def save_config(self, agent_id: str, config: Dict[str, Any]) -> None:
        """Save configuration for a specific agent.
        
        Args:
            agent_id: ID of the agent to save configuration for.
            config: Configuration dictionary to save.
        """
        with self._lock:
            config_file = os.path.join(self._config_path, f"{agent_id}.json")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            self._configs[agent_id] = config
    
    def update_config(self, agent_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration for a specific agent.
        
        Args:
            agent_id: ID of the agent to update configuration for.
            updates: Dictionary containing configuration updates.
            
        Returns:
            The updated configuration dictionary.
        """
        with self._lock:
            config = self._configs.get(agent_id, {})
            config.update(updates)
            self.save_config(agent_id, config)
            return config
    
    def get_config(self, agent_id: str) -> Dict[str, Any]:
        """Get the current configuration for a specific agent.
        
        Args:
            agent_id: ID of the agent to get configuration for.
            
        Returns:
            The current configuration dictionary.
        """
        with self._lock:
            return self._configs.get(agent_id, {})
    
    def delete_config(self, agent_id: str) -> None:
        """Delete configuration for a specific agent.
        
        Args:
            agent_id: ID of the agent to delete configuration for.
        """
        with self._lock:
            config_file = os.path.join(self._config_path, f"{agent_id}.json")
            if os.path.exists(config_file):
                os.remove(config_file)
            self._configs.pop(agent_id, None)