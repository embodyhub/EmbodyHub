"""Multi-Agent Coordination Module for EmbodyHub

This module implements coordination mechanisms for multi-agent systems,
enabling efficient communication and collaboration between agents.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from .agent import Agent
from .adaptive_learning import AdaptiveLearningManager

class MessageType(Enum):
    """Types of messages that can be exchanged between agents."""
    STATE_UPDATE = "state_update"
    ACTION_REQUEST = "action_request"
    TASK_DELEGATION = "task_delegation"
    PERFORMANCE_FEEDBACK = "performance_feedback"

@dataclass
class AgentMessage:
    """Container for inter-agent communication messages."""
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float

class MultiAgentCoordinator:
    """Manages coordination and communication between multiple agents.
    
    This class implements mechanisms for agent collaboration, task delegation,
    and shared learning in multi-agent systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the multi-agent coordinator.
        
        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.agents: Dict[str, Agent] = {}
        self.message_queue: List[AgentMessage] = []
        self.learning_manager = AdaptiveLearningManager()
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize coordination components."""
        self.max_queue_size = self.config.get('max_queue_size', 1000)
        self.collaboration_threshold = self.config.get('collaboration_threshold', 0.7)
    
    def register_agent(self, agent_id: str, agent: Agent) -> None:
        """Register an agent with the coordinator.
        
        Args:
            agent_id: Unique identifier for the agent.
            agent: The agent instance to register.
        """
        if agent_id in self.agents:
            raise ValueError(f"Agent with ID {agent_id} already registered")
        self.agents[agent_id] = agent
    
    def send_message(self, message: AgentMessage) -> bool:
        """Send a message between agents.
        
        Args:
            message: The message to send.
            
        Returns:
            True if message was successfully queued, False otherwise.
        """
        if len(self.message_queue) >= self.max_queue_size:
            return False
            
        if message.sender_id not in self.agents or \
           message.receiver_id not in self.agents:
            return False
            
        self.message_queue.append(message)
        return True
    
    def process_messages(self) -> None:
        """Process all pending messages in the queue."""
        while self.message_queue:
            message = self.message_queue.pop(0)
            self._handle_message(message)
    
    def _handle_message(self, message: AgentMessage) -> None:
        """Handle a single message based on its type.
        
        Args:
            message: The message to handle.
        """
        if message.message_type == MessageType.STATE_UPDATE:
            self._handle_state_update(message)
        elif message.message_type == MessageType.ACTION_REQUEST:
            self._handle_action_request(message)
        elif message.message_type == MessageType.TASK_DELEGATION:
            self._handle_task_delegation(message)
        elif message.message_type == MessageType.PERFORMANCE_FEEDBACK:
            self._handle_performance_feedback(message)
    
    def _handle_state_update(self, message: AgentMessage) -> None:
        """Handle state update messages between agents.
        
        Args:
            message: The state update message.
        """
        receiver = self.agents[message.receiver_id]
        if hasattr(receiver, 'update_shared_state'):
            receiver.update_shared_state(message.content)
    
    def _handle_action_request(self, message: AgentMessage) -> None:
        """Handle action request messages between agents.
        
        Args:
            message: The action request message.
        """
        receiver = self.agents[message.receiver_id]
        if hasattr(receiver, 'process_action_request'):
            receiver.process_action_request(message.content)
    
    def _handle_task_delegation(self, message: AgentMessage) -> None:
        """Handle task delegation messages between agents.
        
        Args:
            message: The task delegation message.
        """
        receiver = self.agents[message.receiver_id]
        if hasattr(receiver, 'accept_delegated_task'):
            receiver.accept_delegated_task(message.content)
    
    def _handle_performance_feedback(self, message: AgentMessage) -> None:
        """Handle performance feedback messages between agents.
        
        Args:
            message: The performance feedback message.
        """
        if 'reward' in message.content and 'loss' in message.content:
            self.learning_manager.update_learning_metrics(
                message.sender_id,
                message.content['reward'],
                message.content['loss']
            )
    
    def optimize_collaboration(self, agent_id: str) -> Dict[str, Any]:
        """Optimize collaboration strategy for an agent.
        
        Args:
            agent_id: ID of the agent to optimize for.
            
        Returns:
            Dictionary containing optimization parameters.
        """
        if agent_id not in self.agents:
            return {}
            
        agent = self.agents[agent_id]
        return self.learning_manager.adjust_learning_strategy(agent_id, agent)