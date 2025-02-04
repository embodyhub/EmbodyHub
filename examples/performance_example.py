from datetime import datetime
from typing import Dict, Any
import time
import random

from src.core.performance_analyzer import PerformanceAnalyzer
from src.core.performance_optimizer import PerformanceOptimizer

def simulate_agent_metrics() -> Dict[str, Any]:
    """Simulate agent performance metrics data."""
    return {
        'response_time': random.uniform(0.1, 2.0),
        'cpu': random.uniform(20, 90),
        'memory': random.uniform(30, 95),
        'gpu': random.uniform(10, 80) if random.random() > 0.3 else None,
        'accuracy': random.uniform(0.8, 0.99)
    }

def main():
    """Demonstrate performance analysis and optimization process."""
    # Initialize performance analyzer and optimizer
    analyzer = PerformanceAnalyzer()
    optimizer = PerformanceOptimizer()

    print("Starting performance analysis and optimization example...\n")

    # Simulate collecting 30 seconds of performance data
    print("Collecting performance metrics...")
    start_time = time.time()
    while time.time() - start_time < 30:
        metrics = simulate_agent_metrics()
        analyzer.collect_metrics(metrics)
        time.sleep(1)  # Collect data every second

    # Analyze performance data
    print("\nAnalyzing performance data...")
    analysis_results = analyzer.analyze_performance()
    
    # Print analysis results summary
    print("\nPerformance Analysis Summary:")
    if 'response_time' in analysis_results:
        rt = analysis_results['response_time']
        print(f"Response Time: Average {rt.get('mean', 0):.2f}s, P95 {rt.get('p95', 0):.2f}s")
    
    if 'resource_usage' in analysis_results:
        ru = analysis_results['resource_usage']
        if 'cpu' in ru:
            print(f"CPU Usage: Average {ru['cpu'].get('mean', 0):.1f}%")
        if 'memory' in ru:
            print(f"Memory Usage: Average {ru['memory'].get('mean', 0):.1f}%")

    # Generate and apply optimization strategies
    print("\nGenerating optimization strategies...")
    optimization_results = optimizer.optimize(analysis_results)
    
    # Print applied optimization strategies
    print("\nApplied Optimization Strategies:")
    for strategy in optimization_results.get('applied_strategies', []):
        strategy_info = strategy['strategy']
        print(f"- Type: {strategy_info['type']}")
        print(f"  Action: {strategy_info['action']}")
        print(f"  Priority: {strategy_info['priority']}")

    print("\nPerformance analysis and optimization example completed")

if __name__ == '__main__':
    main()