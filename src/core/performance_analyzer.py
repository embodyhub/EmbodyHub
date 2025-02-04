import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os
import logging

class PerformanceAnalyzer:
    """Analyzer for AI agent performance metrics."""

    def __init__(self, metrics_config: Optional[Dict[str, Any]] = None):
        """Initialize performance analyzer.

        Args:
            metrics_config: Configuration for metrics collection and analysis.
        """
        self.metrics_config = metrics_config or {}
        self.metrics_history: List[Dict[str, Any]] = []
        self.analysis_results: Dict[str, Any] = {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for performance analysis."""
        log_dir = 'logs/performance'
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('PerformanceAnalyzer')

    def collect_metrics(self, metrics: Dict[str, Any]) -> None:
        """Collect performance metrics.

        Args:
            metrics: Performance metrics data point.
        """
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'data': metrics
        }
        self.metrics_history.append(metrics_entry)
        self.logger.info(f"Collected metrics: {metrics}")

    def analyze_performance(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """Analyze collected performance metrics.

        Args:
            time_window: Time window in seconds for analysis, None for all data.

        Returns:
            Analysis results.
        """
        if not self.metrics_history:
            return {}

        # Filter metrics by time window if specified
        metrics_data = self._filter_metrics_by_time(time_window)
        if not metrics_data:
            return {}

        analysis = {
            'response_time': self._analyze_response_time(metrics_data),
            'resource_usage': self._analyze_resource_usage(metrics_data),
            'accuracy': self._analyze_accuracy(metrics_data),
            'throughput': self._analyze_throughput(metrics_data),
            'timestamp': datetime.now().isoformat()
        }

        self.analysis_results = analysis
        self._generate_performance_report(analysis)
        return analysis

    def _filter_metrics_by_time(self, time_window: Optional[int]) -> List[Dict[str, Any]]:
        """Filter metrics by time window.

        Args:
            time_window: Time window in seconds.

        Returns:
            Filtered metrics data.
        """
        if not time_window:
            return [m['data'] for m in self.metrics_history]

        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        return [
            m['data'] for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) >= cutoff_time
        ]

    def _analyze_response_time(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze response time metrics.

        Args:
            metrics_data: List of metrics data points.

        Returns:
            Response time analysis results.
        """
        response_times = [
            m.get('response_time', 0) for m in metrics_data
            if 'response_time' in m
        ]

        if not response_times:
            return {}

        return {
            'mean': float(np.mean(response_times)),
            'median': float(np.median(response_times)),
            'std': float(np.std(response_times)),
            'p95': float(np.percentile(response_times, 95)),
            'p99': float(np.percentile(response_times, 99))
        }

    def _analyze_resource_usage(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze resource usage metrics.

        Args:
            metrics_data: List of metrics data points.

        Returns:
            Resource usage analysis results.
        """
        cpu_usage = []
        memory_usage = []
        gpu_usage = []

        for m in metrics_data:
            if 'cpu' in m:
                cpu_usage.append(m['cpu'])
            if 'memory' in m:
                memory_usage.append(m['memory'])
            if 'gpu' in m:
                gpu_usage.append(m['gpu'])

        result = {}
        if cpu_usage:
            result['cpu'] = {
                'mean': float(np.mean(cpu_usage)),
                'max': float(np.max(cpu_usage)),
                'utilization': float(np.mean(cpu_usage))
            }
        if memory_usage:
            result['memory'] = {
                'mean': float(np.mean(memory_usage)),
                'max': float(np.max(memory_usage)),
                'utilization': float(np.mean(memory_usage))
            }
        if gpu_usage:
            result['gpu'] = {
                'mean': float(np.mean(gpu_usage)),
                'max': float(np.max(gpu_usage)),
                'utilization': float(np.mean(gpu_usage))
            }

        return result

    def _analyze_accuracy(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze accuracy metrics.

        Args:
            metrics_data: List of metrics data points.

        Returns:
            Accuracy analysis results.
        """
        accuracy_scores = [
            m.get('accuracy', 0) for m in metrics_data
            if 'accuracy' in m
        ]

        if not accuracy_scores:
            return {}

        return {
            'mean': float(np.mean(accuracy_scores)),
            'std': float(np.std(accuracy_scores)),
            'min': float(np.min(accuracy_scores)),
            'max': float(np.max(accuracy_scores))
        }

    def _analyze_throughput(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze system throughput.

        Args:
            metrics_data: List of metrics data points.

        Returns:
            Throughput analysis results.
        """
        if len(metrics_data) < 2:
            return {}

        # Calculate requests per second
        time_diff = (
            datetime.fromisoformat(self.metrics_history[-1]['timestamp']) -
            datetime.fromisoformat(self.metrics_history[0]['timestamp'])
        ).total_seconds()

        if time_diff <= 0:
            return {}

        requests = len(metrics_data)
        throughput = requests / time_diff

        return {
            'requests_per_second': float(throughput),
            'total_requests': requests,
            'time_period': float(time_diff)
        }

    def _generate_performance_report(self, analysis: Dict[str, Any]) -> None:
        """Generate detailed performance analysis report.

        Args:
            analysis: Performance analysis results.
        """
        report = {
            'analysis': analysis,
            'metrics_count': len(self.metrics_history),
            'time_range': {
                'start': self.metrics_history[0]['timestamp'] if self.metrics_history else None,
                'end': self.metrics_history[-1]['timestamp'] if self.metrics_history else None
            },
            'recommendations': self._generate_recommendations(analysis)
        }

        # Save report to file
        report_dir = 'reports/performance'
        os.makedirs(report_dir, exist_ok=True)
        report_path = f'{report_dir}/performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance report generated: {report_path}")

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate performance optimization recommendations.

        Args:
            analysis: Performance analysis results.

        Returns:
            List of recommendations.
        """
        recommendations = []

        # Response time recommendations
        if 'response_time' in analysis:
            rt = analysis['response_time']
            if rt.get('p95', 0) > 1.0:  # Assume target P95 response time is 1 second
                recommendations.append({
                    'category': 'response_time',
                    'severity': 'high',
                    'message': 'Optimize response time, current P95 latency exceeds target value'
                })

        # Resource usage recommendations
        if 'resource_usage' in analysis:
            ru = analysis['resource_usage']
            if 'cpu' in ru and ru['cpu'].get('utilization', 0) > 80:
                recommendations.append({
                    'category': 'resource_usage',
                    'severity': 'medium',
                    'message': 'High CPU utilization, consider resource scaling or optimizing compute-intensive operations'
                })

        # Accuracy recommendations
        if 'accuracy' in analysis:
            acc = analysis['accuracy']
            if acc.get('mean', 0) < 0.95:  # Assume target accuracy is 95%
                recommendations.append({
                    'category': 'accuracy',
                    'severity': 'high',
                    'message': 'Model accuracy below target value, review training process or increase training data'
                })

        return recommendations