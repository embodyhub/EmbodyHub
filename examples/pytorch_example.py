"""PyTorch Integration Example

This example demonstrates how to use the PyTorch adapter to integrate a simple
reinforcement learning model with EmbodyHub.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

from embodyhub.core.environment import Environment
from embodyhub.adapters.pytorch_adapter import PyTorchAdapter

# Define a simple policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Create a simple environment
class SimpleEnvironment(Environment):
    def _initialize(self) -> None:
        self.state = np.zeros(4)
        self.step_count = 0
        
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self.state = np.random.randn(4)  # Simplified state transition
        self.step_count += 1
        reward = float(action.argmax() == self.state.argmax())  # Simple reward
        done = self.step_count >= 100
        return self.state, reward, done, {}
    
    def reset(self) -> np.ndarray:
        self.state = np.zeros(4)
        self.step_count = 0
        return self.state
    
    def render(self, mode: str = 'human') -> None:
        print(f"State: {self.state}, Step: {self.step_count}")

def main():
    # Create environment and model
    env = SimpleEnvironment()
    model = PolicyNetwork(input_size=4, output_size=4)
    
    # Initialize PyTorch adapter
    adapter = PyTorchAdapter(config={'device': 'cpu'})
    adapter.register_model('policy', model)
    
    # Run a simple episode
    obs = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Convert observation to PyTorch tensor
        torch_obs = adapter.convert_observation(obs)
        
        # Get action from model
        with torch.no_grad():
            policy = adapter.get_model('policy')
            action_probs = policy(torch_obs)
            action = adapter.convert_action(action_probs)
        
        # Take step in environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        env.render()
    
    print(f"Episode finished with total reward: {total_reward}")

if __name__ == '__main__':
    main()