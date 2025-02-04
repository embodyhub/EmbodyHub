"""Performance Profiling Tools for EmbodyHub

This module provides comprehensive profiling tools for analyzing and monitoring
the performance of embodied AI applications.
"""

from typing import Any, Dict, Optional, List, Tuple
import time
import psutil
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class ProfilingResult:
    """Container for profiling results."""
    execution_time: float
    memory_usage: Dict[str, float]
    gpu_usage: Optional[Dict[str, float]] = None
    cpu_usage: float = 0.0

class ModelProfiler:
    """Profiler for analyzing model inference performance."""
    
    def __init__(self, warmup_runs: int = 5, num_runs: int = 100):
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
        
    def profile_inference(self, model: torch.nn.Module, 
                         sample_input: torch.Tensor) -> ProfilingResult:
        """Profile model inference performance.
        
        Args:
            model: The PyTorch model to profile.
            sample_input: Sample input tensor for the model.
            
        Returns:
            ProfilingResult containing performance metrics.
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                model(sample_input)
        
        # Measure execution time
        start_time = time.time()
        for _ in range(self.num_runs):
            with torch.no_grad():
                model(sample_input)
        avg_time = (time.time() - start_time) / self.num_runs
        
        # Memory usage
        memory_usage = {
            'allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }
        
        # GPU usage
        gpu_usage = None
        if torch.cuda.is_available():
            gpu_usage = {
                'utilization': torch.cuda.utilization(),
                'power_usage': torch.cuda.power_usage()
            }
        
        return ProfilingResult(
            execution_time=avg_time,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            cpu_usage=psutil.cpu_percent()
        )

class MemoryProfiler:
    """Profiler for analyzing memory usage patterns."""
    
    def profile_memory(self) -> Dict[str, float]:
        """Profile current memory usage.
        
        Returns:
            Dictionary containing memory usage statistics.
        """
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024 ** 3),  # GB
            'available': memory.available / (1024 ** 3),
            'used': memory.used / (1024 ** 3),
            'percent': memory.percent
        }
    
    def track_tensor_allocations(self) -> None:
        """Enable tracking of PyTorch tensor allocations."""
        torch.autograd.set_detect_anomaly(True)
        
    def get_tensor_statistics(self) -> Dict[str, int]:
        """Get statistics about PyTorch tensor allocations.
        
        Returns:
            Dictionary containing tensor allocation statistics.
        """
        return {
            'num_tensors': len(torch._storage_classes),
            'num_parameters': sum(p.numel() for p in torch.alloc_params())
        }

class SystemProfiler:
    """Profiler for monitoring system resource usage."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource usage metrics.
        
        Returns:
            Dictionary containing system resource metrics.
        """
        return {
            'cpu_usage': psutil.cpu_percent(percpu=True),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict(),
            'gpu_info': self._get_gpu_info() if torch.cuda.is_available() else None
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU-related information.
        
        Returns:
            Dictionary containing GPU metrics.
        """
        return {
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'memory_allocated': torch.cuda.memory_allocated(),
            'memory_reserved': torch.cuda.memory_reserved()
        }