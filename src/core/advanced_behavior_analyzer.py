import subprocess
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import and install numpy if needed
try:
    import numpy as np
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
        import numpy as np
    except subprocess.CalledProcessError as e:
        print(f"Failed to install numpy: {str(e)}")
        raise ImportError("Failed to import or install numpy module")
    except ImportError as e:
        print(f"Failed to import numpy after installation: {str(e)}")
        raise

# Import and install scikit-learn if needed
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
    except subprocess.CalledProcessError as e:
        print(f"Failed to install scikit-learn: {str(e)}")
        raise ImportError("Failed to import or install scikit-learn module")
    except ImportError as e:
        print(f"Failed to import scikit-learn after installation: {str(e)}")
        raise

from .behavior_analyzer import BehaviorAnalyzer

class AdvancedBehaviorAnalyzer(BehaviorAnalyzer):
    """Enhanced behavior analyzer with advanced analysis and prediction capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced behavior analyzer.

        Args:
            config: Configuration for advanced analysis and prediction.
        """
        super().__init__(config)
        self.trend_predictor = None
        self.behavior_embeddings = {}
        self.long_term_patterns = {}
        self.data_processor = None
        self._initialize_data_processor()

    def _initialize_data_processor(self) -> None:
        """Initialize data processor."""
        from .advanced_data_processor import AdvancedDataProcessor;
        self.data_processor = AdvancedDataProcessor(self.config)
        from .advanced_data_processor import AdvancedDataProcessor
        self.data_processor = AdvancedDataProcessor(self.config)

    def analyze_long_term_patterns(self, window_size: int = 100) -> Dict[str, Any]:
        """Analyze long-term behavior patterns and trends.

        Args:
            window_size: Analysis window size.

        Returns:
            Long-term behavior pattern analysis results.
        """
        # Process behavior data using data processor
        processed_data = self.data_processor.process_behavior_data(self.behavior_history)

        # Calculate long-term trend indicators
        trend_indicators = self._calculate_trend_indicators(processed_data)

        # Analyze behavior stability
        stability_metrics = self._analyze_stability_metrics(processed_data)

        # Generate long-term predictions
        predictions = self._generate_predictions(processed_data)

        # Detect anomalous behaviors
        anomalies = self._detect_anomalies(processed_data)

        return {
            'trends': trend_indicators,
            'stability': stability_metrics,
            'predictions': predictions,
            'anomalies': anomalies
        }

    def _calculate_trend_indicators(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term behavior indicators."""
        # Extract key metrics
        metrics = {
            'reward': self._calculate_trend(data['reward_history']),
            'response_time': self._calculate_trend(data['response_time_history'])
        }

        # Calculate trends and volatility
        return {
            'reward_trend': metrics['reward'],
            'efficiency_trend': -metrics['response_time'],
            'improvement_rate': self._calculate_improvement_rate(metrics)
        }

    def _analyze_stability_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze behavior stability indicators."""
        # Calculate behavior transition frequency
        transitions = np.diff([hash(str(state)) % 10 for state in data['state_history']])
        transition_rate = np.mean(np.abs(transitions))

        # Calculate stability metrics
        return {
            'pattern_consistency': self._calculate_pattern_consistency(data),
            'transition_stability': 1.0 / (1.0 + transition_rate),
            'prediction_confidence': self._calculate_prediction_confidence(data)
        }

    def _generate_predictions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate future behavior predictions."""
        if not self.prediction_model:
            self._train_trend_model(data)

        # Predict future trends
        future_steps = 5  # Predict next 5 time steps
        predictions = []
        current_features = self._extract_prediction_features(data)

        for _ in range(future_steps):
            pred = self.prediction_model.predict([current_features])[0]
            predictions.append(float(pred))
            # Update features for next step prediction
            current_features = self._update_prediction_features(current_features, pred)

        return {
            'predictions': predictions,
            'confidence': self._calculate_prediction_confidence(data)
        }

    def _train_trend_model(self, data: Dict[str, Any]) -> None:
        """Train trend prediction model."""
        # Prepare training data
        features = self._extract_prediction_features(data)
        y = features[1:, -1]  # Use next time step's reward as target

        # Train model
        self.prediction_model.fit(features[:-1], y)

    def _extract_prediction_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features for trend prediction."""
        return np.array([
            data['reward_history'],
            data['response_time_history'],
            data['confidence_history']
        ]).T

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate numerical sequence trend."""
        if len(values) < 2:
            return 0.0
        return np.polyfit(range(len(values)), values, 1)[0]

    def _calculate_improvement_rate(self, metrics: Dict[str, float]) -> float:
        """Calculate improvement rate."""
        return np.mean([metrics['reward_trend'], -metrics['response_time']])

    def _calculate_pattern_consistency(self, data: Dict[str, Any]) -> float:
        """Calculate behavior consistency score."""
        if not data['pattern_history']:
            return 0.0

        # Calculate entropy of behavior distribution
        # 使用字典来统计模式出现次数
        pattern_counts = {}
        for pattern in data['pattern_history']:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        total = len(data['pattern_history'])
        probabilities = [count/total for count in pattern_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities)

        # Normalize consistency score
        max_entropy = np.log2(len(pattern_counts))
        return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate numerical sequence volatility."""
        if len(values) < 2:
            return 0.0
        return np.std(values) / (np.mean(values) + 1e-6)

    def _calculate_pattern_stability(self, patterns: List[str]) -> float:
        """Calculate pattern consistency."""
        if len(patterns) < 2:
            return 1.0

        # Calculate transition matrix
        unique_patterns = list(set(patterns))
        n_patterns = len(unique_patterns)
        transitions = np.zeros((n_patterns, n_patterns))

        for i in range(len(patterns)-1):
            from_idx = unique_patterns.index(patterns[i])
            to_idx = unique_patterns.index(patterns[i+1])
            transitions[from_idx, to_idx] += 1

        # Calculate variance of transition probabilities as consistency indicator
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_probs = transitions / (row_sums + 1e-10)
        return 1.0 - float(np.var(transition_probs))

    def _calculate_prediction_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate prediction confidence."""
        if not self.prediction_model:
            return 0.0

        # Calculate confidence based on prediction stability
        recent_predictions = data.get('recent_predictions', [])
        if len(recent_predictions) < 2:
            return 0.5

        return 1.0 / (1.0 + self._calculate_volatility(recent_predictions))

    def _detect_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalous behavior patterns."""
        if not data or 'processed_features' not in data:
            return []

        # Use data processor to detect anomalies
        anomaly_scores = self.data_processor.detect_anomalies(data['processed_features'])

        # Analyze anomalous behavior characteristics
        anomalies = []
        for i, score in enumerate(anomaly_scores):
            if score < 0:  # Anomaly detected
                anomalies.append({
                    'index': i,
                    'score': float(score),
                    'timestamp': data['timestamps'][i],
                    'features': data['processed_features'][i].tolist()
                })

        return anomalies