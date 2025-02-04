"""Example demonstrating the use of EmbodyHub's performance optimization tools.

This example shows how to use various optimization techniques to improve
the performance of embodied AI applications.
"""

import torch
import torch.nn as nn
from src.core.optimizer import ModelOptimizer, MemoryManager, ParallelExecutor

# Define a simple neural network for demonstration
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def main():
    # Create a sample model
    model = SimpleModel()
    
    # Initialize optimizers
    model_optimizer = ModelOptimizer({
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })
    memory_manager = MemoryManager()
    parallel_executor = ParallelExecutor({'num_workers': 4})
    
    # Optimize model for inference
    print("Optimizing model...")
    quantized_model = model_optimizer.quantize_model(model)
    model_optimizer.optimize_memory(quantized_model)
    
    # Demonstrate memory optimization
    print("\nDemonstrating memory optimization...")
    sample_tensor = torch.randn(1000, 1000, dtype=torch.float32)
    optimized_tensor = memory_manager.optimize_tensor(sample_tensor)
    print(f"Original tensor dtype: {sample_tensor.dtype}")
    print(f"Optimized tensor dtype: {optimized_tensor.dtype}")
    
    # Demonstrate parallel execution
    print("\nDemonstrating parallel execution...")
    def process_data(x):
        return x * 2
    
    data = list(range(1000))
    results = parallel_executor.parallel_map(process_data, data)
    print(f"Processed {len(results)} items in parallel")

if __name__ == '__main__':
    main()