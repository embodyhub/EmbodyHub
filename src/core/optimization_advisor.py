from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import json
import os
import logging
from .performance_analyzer import PerformanceAnalyzer
from .behavior_analyzer import BehaviorAnalyzer

class OptimizationAdvisor:
    """Advisor for generating optimization suggestions based on agent analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize optimization advisor.

        Args:
            config: Configuration for optimization advice generation.
        """
        self.config = config or {}
        self.performance_analyzer = PerformanceAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for optimization advisor."""
        log_dir = 'logs/advisor'
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/advisor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('OptimizationAdvisor')

    def generate_optimization_advice(self, 
                                    performance_data: Dict[str, Any],
                                    behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive optimization advice.

        Args:
            performance_data: Performance metrics and analysis.
            behavior_data: Behavior patterns and analysis.

        Returns:
            Optimization advice and visualization data.
        """
        # Analyze performance data
        self.performance_analyzer.collect_metrics(performance_data)
        performance_analysis = self.performance_analyzer.analyze_performance()

        # Analyze behavior data
        self.behavior_analyzer.record_behavior(behavior_data)
        behavior_patterns = self.behavior_analyzer.identify_patterns()

        # Generate optimization advice
        advice = {
            'timestamp': datetime.now().isoformat(),
            'performance_recommendations': self._generate_performance_advice(performance_analysis),
            'behavior_recommendations': self._generate_behavior_advice(behavior_patterns),
            'visualization_data': self._prepare_visualization_data(performance_analysis, behavior_patterns)
        }

        self._save_advice_report(advice)
        return advice

    def _generate_performance_advice(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Analyze response time
        if 'response_time' in analysis:
            rt_data = analysis['response_time']
            if rt_data.get('p95', 0) > 1.0:
                recommendations.append({
                    'category': 'response_time',
                    'severity': 'high',
                    'message': 'Response time P95 exceeds 1 second, optimize compute-intensive operations or consider caching'
                })

        # Analyze resource usage
        if 'resource_usage' in analysis:
            ru_data = analysis['resource_usage']
            if ru_data.get('memory_usage', 0) > 80:
                recommendations.append({
                    'category': 'resource_usage',
                    'severity': 'medium',
                    'message': 'High memory usage, optimize memory management or increase resource allocation'
                })

        return recommendations

    def _generate_behavior_advice(self, patterns: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate behavior optimization recommendations."""
        recommendations = []

        # Analyze pattern distribution
        if patterns:
            pattern_distribution = [p['percentage'] for p in patterns.values()]
            if max(pattern_distribution) > 70:
                recommendations.append({
                    'category': 'behavior_diversity',
                    'severity': 'medium',
                    'message': 'Behavior patterns too concentrated, increase exploratory behavior to improve adaptability'
                })

        return recommendations

    def _prepare_visualization_data(self,
                                   performance_analysis: Dict[str, Any],
                                   behavior_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for visualization."""
        return {
            'performance_metrics': {
                'response_time_trend': self._extract_time_series_data(
                    self.performance_analyzer.metrics_history, 'response_time'
                ),
                'resource_usage_trend': self._extract_time_series_data(
                    self.performance_analyzer.metrics_history, 'cpu'
                )
            },
            'behavior_metrics': {
                'pattern_distribution': [
                    {
                        'pattern': name,
                        'percentage': data['percentage']
                    }
                    for name, data in behavior_patterns.items()
                ]
            }
        }

    def _extract_time_series_data(self,
                                 history: List[Dict[str, Any]],
                                 metric_key: str) -> List[Dict[str, Any]]:
        """Extract time series data for visualization."""
        return [
            {
                'timestamp': entry['timestamp'],
                'value': entry['data'].get(metric_key, 0)
            }
            for entry in history
            if metric_key in entry['data']
        ]

    def _save_advice_report(self, advice: Dict[str, Any]) -> None:
        """Save optimization advice report."""
        report_dir = 'reports/optimization'
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = f'{report_dir}/optimization_advice_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(advice, f, indent=2)
        
        self.logger.info(f"Optimization advice report saved: {report_path}")