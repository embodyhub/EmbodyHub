"""Agent Manager Module

This module provides functionality for managing the lifecycle and state of agents
in the EmbodyHub framework.
"""

from typing import Dict, List, Optional, Any
from threading import Lock
from .agent import Agent
from .system_monitor import SystemMonitor

class AgentManager:
    """Manages the lifecycle and state of agents in the system.
    
    This class handles agent creation, monitoring, and optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent manager.
        
        Args:
            config: Optional configuration dictionary for the manager.
        """
        self.config = config or {}
        self._agents: Dict[str, Agent] = {}
        self._agent_states: Dict[str, str] = {}
        self._lock = Lock()
        self._system_monitor = SystemMonitor()
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize manager components."""
        self._system_monitor.start()
    
    def register_agent(self, agent_id: str, agent: Agent) -> None:
        """Register a new agent with the manager.
        
        Args:
            agent_id: Unique identifier for the agent.
            agent: The agent instance to register.
        """
        with self._lock:
            if agent_id in self._agents:
                raise ValueError(f"Agent with ID {agent_id} already exists")
            self._agents[agent_id] = agent
            self._agent_states[agent_id] = 'initialized'
    
    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the manager.
        
        Args:
            agent_id: ID of the agent to remove.
        """
        with self._lock:
            if agent_id not in self._agents:
                raise ValueError(f"Agent with ID {agent_id} not found")
            del self._agents[agent_id]
            del self._agent_states[agent_id]
    
    def get_agent(self, agent_id: str) -> Agent:
        """Retrieve an agent by its ID.
        
        Args:
            agent_id: ID of the agent to retrieve.
            
        Returns:
            The requested agent instance.
        """
        with self._lock:
            if agent_id not in self._agents:
                raise ValueError(f"Agent with ID {agent_id} not found")
            return self._agents[agent_id]
    
    def get_agent_state(self, agent_id: str) -> str:
        """Get the current state of an agent.
        
        Args:
            agent_id: ID of the agent.
            
        Returns:
            Current state of the agent.
        """
        with self._lock:
            if agent_id not in self._agent_states:
                raise ValueError(f"Agent with ID {agent_id} not found")
            return self._agent_states[agent_id]
    
    def update_agent_state(self, agent_id: str, state: str) -> None:
        """Update the state of an agent.
        
        Args:
            agent_id: ID of the agent.
            state: New state to set.
        """
        with self._lock:
            if agent_id not in self._agent_states:
                raise ValueError(f"Agent with ID {agent_id} not found")
            self._agent_states[agent_id] = state
    
    def get_all_agents(self) -> List[str]:
        """Get a list of all registered agent IDs.
        
        Returns:
            List of agent IDs.
        """
        with self._lock:
            return list(self._agents.keys())
    
    def reset_all_agents(self) -> None:
        """Reset all registered agents to their initial state."""
        with self._lock:
            for agent in self._agents.values():
                agent.reset()
                
    def shutdown(self) -> None:
        """Shutdown the manager and cleanup resources."""
        self._system_monitor.stop()
        self.reset_all_agents()