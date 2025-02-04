"""System Resource Monitoring Module

This module provides functionality for monitoring and collecting system resource metrics
in real-time, including CPU, memory, disk, and GPU usage.
"""

from typing import Dict, Any, Optional
import time
import threading
from queue import Queue
import psutil
import torch

class SystemMonitor:
    """System resource monitor for collecting real-time performance metrics."""
    
    def __init__(self, sampling_interval: float = 1.0):
        """Initialize the system monitor.
        
        Args:
            sampling_interval: Time interval between metric collections in seconds.
        """
        self.sampling_interval = sampling_interval
        self._metrics_queue: Queue = Queue(maxsize=100)
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the system monitoring process."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._collect_metrics,
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop(self) -> None:
        """Stop the system monitoring process."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the most recent system metrics.
        
        Returns:
            Dictionary containing the latest system metrics.
        """
        try:
            return self._metrics_queue.get_nowait()
        except:
            return self._collect_system_metrics()
    
    def _collect_metrics(self) -> None:
        """Continuously collect system metrics."""
        while self._running:
            metrics = self._collect_system_metrics()
            
            # Update metrics queue
            if self._metrics_queue.full():
                try:
                    self._metrics_queue.get_nowait()
                except:
                    pass
            self._metrics_queue.put(metrics)
            
            time.sleep(self.sampling_interval)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system resource metrics.
        
        Returns:
            Dictionary containing various system metrics.
        """
        metrics = {
            'timestamp': time.time(),
            'cpu': {
                'percent': psutil.cpu_percent(percpu=True),
                'count': psutil.cpu_count(),
                'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            'memory': {
                'virtual': psutil.virtual_memory()._asdict(),
                'swap': psutil.swap_memory()._asdict()
            },
            'disk': {
                'usage': psutil.disk_usage('/')._asdict(),
                'io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None
            },
            'network': {
                'io': psutil.net_io_counters()._asdict(),
                'connections': len(psutil.net_connections())
            }
        }
        
        # Add GPU metrics if available
        if torch.cuda.is_available():
            metrics['gpu'] = self._collect_gpu_metrics()
        
        return metrics
    
    def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect GPU-related metrics.
        
        Returns:
            Dictionary containing GPU metrics.
        """
        return {
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'memory': {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved()
            },
            'utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
        }