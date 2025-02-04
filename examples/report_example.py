import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.behavior_analyzer import BehaviorAnalyzer
from src.core.performance_analyzer import PerformanceAnalyzer
from src.core.report_generator import ReportGenerator
from datetime import datetime
import random
import time

def simulate_agent_behavior():
    """Simulate agent behavior data."""
    return {
        'action_value': random.uniform(0, 1),
        'confidence': random.uniform(0.5, 1.0),
        'response_time': random.uniform(0.1, 2.0),
        'action_type': random.choice(['explore', 'exploit', 'learn']),
        'reward': random.uniform(-1, 1)
    }

def simulate_performance_metrics():
    """Simulate performance metrics data."""
    return {
        'response_time': random.uniform(0.1, 2.0),
        'cpu': random.uniform(20, 90),
        'memory': random.uniform(30, 95),
        'gpu': random.uniform(10, 80) if random.random() > 0.3 else None,
        'accuracy': random.uniform(0.8, 0.99)
    }

def main():
    # Initialize analyzers and report generator
    behavior_analyzer = BehaviorAnalyzer()
    performance_analyzer = PerformanceAnalyzer()
    report_generator = ReportGenerator()

    print("Starting to collect agent data...")
    # Simulate data collection
    for _ in range(50):
        behavior_data = simulate_agent_behavior()
        performance_data = simulate_performance_metrics()
        
        behavior_analyzer.record_behavior(behavior_data)
        performance_analyzer.collect_metrics(performance_data)
        time.sleep(0.1)

    # Generate behavior analysis report
    print("\nGenerating behavior analysis report...")
    behavior_report = behavior_analyzer.generate_analysis_report()
    behavior_report_path = report_generator.generate_behavior_report(behavior_report, 'html')
    print(f"Behavior analysis report saved: {behavior_report_path}")

    # Generate performance analysis report
    print("\nGenerating performance analysis report...")
    performance_report = performance_analyzer.analyze_performance()
    performance_report_path = report_generator.generate_performance_report(performance_report, 'html')
    print(f"Performance analysis report saved: {performance_report_path}")

    # Generate combined analysis report
    print("\nGenerating combined analysis report...")
    combined_report_path = report_generator.generate_combined_report(
        behavior_report,
        performance_report,
        'html'
    )
    print(f"Combined analysis report saved: {combined_report_path}")

    # Export CSV format data
    print("\nExporting CSV format data...")
    csv_behavior_path = report_generator.generate_behavior_report(behavior_report, 'csv')
    csv_performance_path = report_generator.generate_performance_report(performance_report, 'csv')
    print(f"CSV format data saved:\n- Behavior data: {csv_behavior_path}\n- Performance data: {csv_performance_path}")

if __name__ == '__main__':
    main()