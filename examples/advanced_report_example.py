import os
import time
import random
from datetime import datetime
from typing import Dict, Any
from src.core.behavior_analyzer import BehaviorAnalyzer
from src.core.performance_analyzer import PerformanceAnalyzer
from src.core.advanced_report_generator import AdvancedReportGenerator

def simulate_agent_behavior() -> Dict[str, Any]:
    """Simulate agent behavior data."""
    return {
        'action_value': random.uniform(0, 1),
        'confidence': random.uniform(0.5, 1.0),
        'response_time': random.uniform(0.1, 2.0),
        'action_type': random.choice(['explore', 'exploit', 'learn']),
        'reward': random.uniform(-1, 1),
        'timestamp': datetime.now().isoformat()
    }

def simulate_performance_metrics() -> Dict[str, Any]:
    """Simulate performance metrics data."""
    return {
        'response_time': random.uniform(0.1, 2.0),
        'cpu_usage': random.uniform(20, 90),
        'memory_usage': random.uniform(30, 95),
        'gpu_usage': random.uniform(10, 80) if random.random() > 0.3 else None,
        'accuracy': random.uniform(0.8, 0.99),
        'timestamp': datetime.now().isoformat()
    }

def main():
    # Initialize analyzers and advanced report generator
    behavior_analyzer = BehaviorAnalyzer()
    performance_analyzer = PerformanceAnalyzer()
    report_generator = AdvancedReportGenerator()

    print("Starting to collect agent data...")
    # Simulate data collection
    for _ in range(50):
        behavior_data = simulate_agent_behavior()
        performance_data = simulate_performance_metrics()
        
        behavior_analyzer.record_behavior(behavior_data)
        performance_analyzer.collect_metrics(performance_data)
        time.sleep(0.1)

    # Generate advanced behavior analysis report
    print("\nGenerating advanced behavior analysis report...")
    behavior_report = behavior_analyzer.generate_analysis_report()
    behavior_report_path = report_generator.generate_advanced_behavior_report(behavior_report)
    print(f"Advanced behavior analysis report saved: {behavior_report_path}")

    # Generate advanced performance analysis report
    print("\nGenerating advanced performance analysis report...")
    performance_report = performance_analyzer.analyze_performance()
    performance_report_path = report_generator.generate_advanced_performance_report(performance_report)
    print(f"Advanced performance analysis report saved: {performance_report_path}")

    # Export reports in different formats
    print("\nExporting reports in different formats...")
    formats = ['json', 'csv', 'html']
    for format in formats:
        b_path = report_generator.generate_advanced_behavior_report(behavior_report, format)
        p_path = report_generator.generate_advanced_performance_report(performance_report, format)
        print(f"{format.upper()} format reports saved:\n- Behavior report: {b_path}\n- Performance report: {p_path}")

if __name__ == '__main__':
    main()