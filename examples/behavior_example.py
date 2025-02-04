import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.behavior_analyzer import BehaviorAnalyzer
import random
from typing import Dict, Any
import time

def simulate_agent_behavior() -> Dict[str, Any]:
    """Simulate agent behavior data."""
    return {
        'action_value': random.uniform(0, 1),
        'confidence': random.uniform(0.5, 1.0),
        'response_time': random.uniform(0.1, 2.0),
        'action_type': random.choice(['explore', 'exploit', 'learn']),
        'reward': random.uniform(-1, 1)
    }

def main():
    # Initialize behavior analyzer
    analyzer = BehaviorAnalyzer()
    print("Starting to collect agent behavior data...")

    # Simulate behavior data collection
    for _ in range(50):
        behavior = simulate_agent_behavior()
        analyzer.record_behavior(behavior)
        time.sleep(0.1)  # Simulate behavior interval

    # Identify behavior patterns
    print("\nAnalyzing behavior patterns...")
    patterns = analyzer.identify_patterns(n_clusters=3)
    print(f"Discovered {len(patterns)} behavior patterns:")
    for pattern_name, pattern_data in patterns.items():
        print(f"- {pattern_name}: Percentage {pattern_data['percentage']:.2f}%")

    # Train prediction model
    print("\nTraining behavior prediction model...")
    training_results = analyzer.train_prediction_model()
    print(f"Model training results: Accuracy = {training_results.get('accuracy', 0):.2f}")

    # Predict next behavior
    print("\nPredicting next behavior...")
    current_state = simulate_agent_behavior()
    prediction = analyzer.predict_next_behavior(current_state)
    if prediction:
        print(f"Prediction result: Pattern {prediction['predicted_pattern']}")
        print(f"Prediction confidence: {prediction['confidence']:.2f}")

    # Generate analysis report
    print("\nGenerating behavior analysis report...")
    report = analyzer.generate_analysis_report()
    print(f"Report saved to: {report_dir}/behavior_report_*.json")

    # Output recommendations
    if report.get('recommendations'):
        print("\nOptimization recommendations:")
        for rec in report['recommendations']:
            print(f"- [{rec['severity']}] {rec['message']}")

if __name__ == '__main__':
    main()