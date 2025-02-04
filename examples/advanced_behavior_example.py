"""Advanced Behavior Analysis Example

This example demonstrates the usage of advanced behavior analysis
and prediction capabilities in EmbodyHub.
"""

import time
from datetime import datetime
from typing import Dict, Any
import random
from src.core.advanced_data_analyzer import AdvancedDataAnalyzer

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

def main():
    # Initialize advanced data analyzer
    analyzer = AdvancedDataAnalyzer()
    behavior_data = []

    print("Starting to collect agent behavior data...")
    # Simulate data collection process
    for _ in range(50):
        data = simulate_agent_behavior()
        behavior_data.append(data)
        time.sleep(0.1)

    # Analyze behavior patterns
    print("\nAnalyzing behavior patterns...")
    pattern_analysis = analyzer.analyze_behavior_patterns(behavior_data)
    print("\nBehavior pattern analysis results:")
    print(f"Number of patterns discovered: {len(pattern_analysis['patterns'])}")
    for pattern_id, pattern_info in pattern_analysis['patterns'].items():
        print(f"\n{pattern_id}:")
        print(f"- Size: {pattern_info['size']}")
        print(f"- Variance: {pattern_info['variance']:.4f}")

    # Predict behavior trends
    print("\nPredicting behavior trends...")
    prediction_result = analyzer.predict_behavior_trend(behavior_data)
    if prediction_result:
        print("\nPrediction results:")
        print(f"Predicted values: {prediction_result['predictions']}")
        print(f"Confidence: {prediction_result['confidence']:.4f}")

    # Analyze temporal trends
    print("\nAnalyzing temporal trends...")
    trends = pattern_analysis['trends']
    print(f"Reward trend: {trends['reward_trend']:.4f}")
    print(f"Analysis period: {trends['period_start']} to {trends['period_end']}")

if __name__ == '__main__':
    main()