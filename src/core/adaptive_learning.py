"""Adaptive Learning Module for EmbodyHub

This module implements adaptive learning mechanisms for agents,
enabling them to dynamically adjust their behavior and learning strategies
based on performance feedback and environmental conditions.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass
from .agent import Agent
from .profiler import ProfilingResult

@dataclass
class LearningMetrics:
    """Container for learning performance metrics."""
    reward_history: List[float]
    loss_history: List[float]
    adaptation_rate: float
    performance_score: float

class AdaptiveLearningManager:
    """Manages adaptive learning strategies for agents.
    
    This class implements mechanisms for dynamically adjusting agent learning
    parameters based on performance metrics and environmental feedback.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the adaptive learning manager.
        
        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.learning_history: Dict[str, LearningMetrics] = {}
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize adaptive learning components."""
        self.adaptation_threshold = self.config.get('adaptation_threshold', 0.1)
        self.history_window = self.config.get('history_window', 100)
        
    def update_learning_metrics(self, agent_id: str, 
                               reward: float, 
                               loss: float) -> None:
        """Update learning metrics for an agent.
        
        Args:
            agent_id: Unique identifier for the agent.
            reward: Current reward value.
            loss: Current loss value.
        """
        if agent_id not in self.learning_history:
            self.learning_history[agent_id] = LearningMetrics(
                reward_history=[],
                loss_history=[],
                adaptation_rate=1.0,
                performance_score=0.0
            )
            
        metrics = self.learning_history[agent_id]
        metrics.reward_history.append(reward)
        metrics.loss_history.append(loss)
        
        # Maintain fixed window size
        if len(metrics.reward_history) > self.history_window:
            metrics.reward_history.pop(0)
            metrics.loss_history.pop(0)
            
        # Update performance score
        metrics.performance_score = self._calculate_performance_score(metrics)
        
    def adjust_learning_strategy(self, agent_id: str, 
                                agent: Agent) -> Dict[str, Any]:
        """Adjust agent's learning strategy based on performance metrics.
        
        Args:
            agent_id: Unique identifier for the agent.
            agent: The agent instance to adjust.
            
        Returns:
            Dictionary containing the adjusted learning parameters.
        """
        if agent_id not in self.learning_history:
            return {}
            
        metrics = self.learning_history[agent_id]
        
        # Calculate trend in performance
        performance_trend = self._analyze_performance_trend(metrics)
        
        # Adjust learning parameters based on performance
        adjustments = self._compute_parameter_adjustments(
            metrics, performance_trend)
            
        # Apply adjustments to agent configuration
        self._apply_adjustments(agent, adjustments)
        
        return adjustments
    
    def _calculate_performance_score(self, 
                                   metrics: LearningMetrics) -> float:
        """Calculate overall performance score from metrics.
        
        Args:
            metrics: Learning metrics for an agent.
            
        Returns:
            Normalized performance score.
        """
        if not metrics.reward_history:
            return 0.0
            
        recent_rewards = metrics.reward_history[-10:]
        recent_losses = metrics.loss_history[-10:]
        
        reward_score = np.mean(recent_rewards)
        loss_score = 1.0 / (1.0 + np.mean(recent_losses))
        
        return 0.7 * reward_score + 0.3 * loss_score
    
    def _analyze_performance_trend(self, 
                                 metrics: LearningMetrics) -> float:
        """Analyze the trend in agent's performance.
        
        Args:
            metrics: Learning metrics for an agent.
            
        Returns:
            Trend coefficient (-1 to 1) indicating performance direction.
        """
        if len(metrics.reward_history) < 2:
            return 0.0
            
        rewards = np.array(metrics.reward_history)
        x = np.arange(len(rewards))
        
        # Calculate linear regression slope
        slope = np.polyfit(x, rewards, 1)[0]
        
        # Normalize slope to [-1, 1] range
        return np.clip(slope * 10, -1, 1)
    
    def _compute_parameter_adjustments(self, 
                                     metrics: LearningMetrics,
                                     trend: float) -> Dict[str, Any]:
        """Compute adjustments to learning parameters.
        
        Args:
            metrics: Learning metrics for an agent.
            trend: Performance trend coefficient.
            
        Returns:
            Dictionary of parameter adjustments.
        """
        adjustments = {}
        
        # Adjust learning rate based on performance trend
        if abs(trend) > self.adaptation_threshold:
            if trend < 0:
                # Decrease learning rate when performance is declining
                adjustments['learning_rate'] = 0.8
            else:
                # Increase learning rate when performance is improving
                adjustments['learning_rate'] = 1.2
                
        # Adjust exploration rate based on performance score
        if metrics.performance_score < 0.3:
            adjustments['exploration_rate'] = 1.5
        elif metrics.performance_score > 0.7:
            adjustments['exploration_rate'] = 0.8
            
        return adjustments
    
    def _apply_adjustments(self, agent: Agent, 
                          adjustments: Dict[str, Any]) -> None:
        """Apply computed adjustments to agent parameters.
        
        Args:
            agent: The agent instance to adjust.
            adjustments: Dictionary of parameter adjustments.
        """
        if hasattr(agent, 'learning_rate'):
            if 'learning_rate' in adjustments:
                agent.learning_rate *= adjustments['learning_rate']
                
        if hasattr(agent, 'exploration_rate'):
            if 'exploration_rate' in adjustments:
                agent.exploration_rate *= adjustments['exploration_rate']