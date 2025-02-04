"""Example demonstrating the use of EmbodyHub's performance profiling tools.

This example shows how to use various profiling tools to analyze and monitor
the performance of embodied AI applications.
"""

import torch
import torch.nn as nn
from src.core.profiler import ModelProfiler, MemoryProfiler, SystemProfiler

# Define a sample model for demonstration
class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc = nn.Linear(32 * 6 * 6, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 6)
        return self.fc(x)

def main():
    # Initialize profilers
    model_profiler = ModelProfiler(warmup_runs=3, num_runs=50)
    memory_profiler = MemoryProfiler()
    system_profiler = SystemProfiler(sampling_interval=1.0)
    
    # Create sample model and input
    model = SampleModel()
    sample_input = torch.randn(1, 3, 32, 32)  # Sample image
    
    print("\n1. Model Inference Profiling")
    print("-" * 30)
    result = model_profiler.profile_inference(model, sample_input)
    print(f"Average inference time: {result.execution_time:.4f} seconds")
    print(f"Memory allocated: {result.memory_usage['allocated'] / 1024**2:.2f} MB")
    if result.gpu_usage:
        print(f"GPU utilization: {result.gpu_usage['utilization']}%")
    print(f"CPU usage: {result.cpu_usage}%")
    
    print("\n2. Memory Usage Analysis")
    print("-" * 30)
    memory_stats = memory_profiler.profile_memory()
    print(f"Total memory: {memory_stats['total']:.2f} GB")
    print(f"Used memory: {memory_stats['used']:.2f} GB")
    print(f"Memory usage: {memory_stats['percent']}%")
    
    # Enable tensor allocation tracking
    memory_profiler.track_tensor_allocations()
    tensor_stats = memory_profiler.get_tensor_statistics()
    print(f"Number of tensors: {tensor_stats['num_tensors']}")
    print(f"Total parameters: {tensor_stats['num_parameters']}")
    
    print("\n3. System Resource Monitoring")
    print("-" * 30)
    system_metrics = system_profiler.get_system_metrics()
    print(f"CPU usage per core: {system_metrics['cpu_usage']}")
    print(f"Disk usage: {system_metrics['disk_usage']}%")
    
    if system_metrics['gpu_info']:
        gpu_info = system_metrics['gpu_info']
        print(f"\nGPU Information:")
        print(f"Number of GPUs: {gpu_info['device_count']}")
        print(f"Current device: {gpu_info['current_device']}")
        print(f"GPU memory allocated: {gpu_info['memory_allocated'] / 1024**2:.2f} MB")

if __name__ == '__main__':
    main()