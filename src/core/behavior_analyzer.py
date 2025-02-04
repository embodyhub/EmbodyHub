import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
import json
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

class BehaviorAnalyzer:
    """Analyzer for AI agent behavior patterns and predictions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize behavior analyzer.

        Args:
            config: Configuration for behavior analysis and prediction.
        """
        self.config = config or {}
        self.behavior_history: List[Dict[str, Any]] = []
        self.pattern_clusters = {}
        self.prediction_model = None
        self.scaler = StandardScaler()
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for behavior analysis."""
        log_dir = 'logs/behavior'
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/behavior_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BehaviorAnalyzer')

    def record_behavior(self, behavior_data: Dict[str, Any]) -> None:
        """Record agent behavior data point.

        Args:
            behavior_data: Behavior metrics and context information.
        """
        behavior_entry = {
            'timestamp': datetime.now().isoformat(),
            'data': behavior_data
        }
        self.behavior_history.append(behavior_entry)
        self.logger.info(f"Recorded behavior: {behavior_data}")

    def identify_patterns(self, n_clusters: int = 3) -> Dict[str, Any]:
        """Identify behavior patterns using clustering.

        Args:
            n_clusters: Number of behavior pattern clusters to identify.

        Returns:
            Identified behavior patterns and their characteristics.
        """
        if len(self.behavior_history) < n_clusters:
            return {}

        # Extract features for clustering
        features = self._extract_behavior_features()
        if not features:
            return {}

        # Normalize features
        scaled_features = self.scaler.fit_transform(features)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)

        # Analyze clusters
        patterns = self._analyze_clusters(cluster_labels, kmeans.cluster_centers_)
        self.pattern_clusters = patterns

        return patterns

    def train_prediction_model(self, test_size: float = 0.2) -> Dict[str, float]:
        """Train a model to predict future behavior patterns.

        Args:
            test_size: Proportion of data to use for testing.

        Returns:
            Model training results and performance metrics.
        """
        if len(self.behavior_history) < 10:
            return {}

        # Prepare features and labels
        features = self._extract_behavior_features()
        if not features:
            return {}

        # Create target labels (next behavior pattern)
        labels = self._create_prediction_labels()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42
        )

        # Train model
        self.prediction_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.prediction_model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.prediction_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results = {
            'accuracy': float(accuracy),
            'test_size': len(y_test),
            'train_size': len(y_train)
        }

        self.logger.info(f"Prediction model trained with results: {results}")
        return results

    def predict_next_behavior(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict agent's next behavior pattern.

        Args:
            current_state: Current agent state and context.

        Returns:
            Predicted behavior pattern and confidence score.
        """
        if not self.prediction_model:
            return {}

        # Extract features from current state
        features = self._extract_state_features(current_state)
        if not features:
            return {}

        # Scale features
        scaled_features = self.scaler.transform([features])

        # Make prediction
        prediction = self.prediction_model.predict(scaled_features)
        probabilities = self.prediction_model.predict_proba(scaled_features)

        prediction_result = {
            'predicted_pattern': int(prediction[0]),
            'confidence': float(np.max(probabilities)),
            'timestamp': datetime.now().isoformat()
        }

        self.logger.info(f"Behavior prediction: {prediction_result}")
        return prediction_result

    def _extract_behavior_features(self) -> np.ndarray:
        """Extract numerical features from behavior history."""
        features = []
        for entry in self.behavior_history:
            feature_vector = self._extract_state_features(entry['data'])
            if feature_vector:
                features.append(feature_vector)
        return np.array(features) if features else np.array([])

    def _extract_state_features(self, state_data: Dict[str, Any]) -> List[float]:
        """Extract features from a single state."""
        features = []
        try:
            # Add relevant numerical features
            if 'action_value' in state_data:
                features.append(float(state_data['action_value']))
            if 'confidence' in state_data:
                features.append(float(state_data['confidence']))
            if 'response_time' in state_data:
                features.append(float(state_data['response_time']))
            if 'reward' in state_data:
                features.append(float(state_data['reward']))
            
            # Add action type encoding
            if 'action_type' in state_data:
                action_types = ['explore', 'exploit', 'learn']
                action_encoding = [1.0 if state_data['action_type'] == t else 0.0 for t in action_types]
                features.extend(action_encoding)
                
            # Add temporal features
            if len(self.behavior_history) > 0:
                time_diff = (datetime.fromisoformat(state_data.get('timestamp', datetime.now().isoformat())) - 
                            datetime.fromisoformat(self.behavior_history[-1]['timestamp'])).total_seconds()
                features.append(time_diff)
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error extracting features: {e}")
            return []
        return features

    def _analyze_clusters(self, labels: np.ndarray, centers: np.ndarray) -> Dict[str, Any]:
        """Analyze behavior pattern clusters with detailed metrics."""
        patterns = {}
        for i in range(len(centers)):
            # Basic statistics
            cluster_mask = labels == i
            cluster_size = int(np.sum(cluster_mask))
            
            # Extract cluster behavior data
            cluster_behaviors = [self.behavior_history[j]['data'] for j in range(len(labels)) if labels[j] == i]
            
            # Calculate cluster behavior features
            rewards = [b.get('reward', 0) for b in cluster_behaviors if 'reward' in b]
            response_times = [b.get('response_time', 0) for b in cluster_behaviors if 'response_time' in b]
            action_types = [b.get('action_type', '') for b in cluster_behaviors if 'action_type' in b]
            
            # Analyze behavior characteristics
            cluster_data = {
                'center': centers[i].tolist(),
                'size': cluster_size,
                'percentage': float(np.mean(cluster_mask) * 100),
                'metrics': {
                    'avg_reward': float(np.mean(rewards)) if rewards else 0,
                    'avg_response_time': float(np.mean(response_times)) if response_times else 0,
                    'action_distribution': {
                        action: action_types.count(action) / len(action_types) * 100
                        for action in set(action_types)
                    } if action_types else {}
                },
                'trend': self._analyze_cluster_trend(cluster_behaviors)
            }
            patterns[f'pattern_{i}'] = cluster_data
        return patterns

    def _analyze_cluster_trend(self, behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze time trends of behavior clusters."""
        if not behaviors:
            return {}

        # Extract time series data
        timestamps = [datetime.fromisoformat(b.get('timestamp', datetime.now().isoformat())) for b in behaviors]
        rewards = [b.get('reward', 0) for b in behaviors]
        response_times = [b.get('response_time', 0) for b in behaviors]

        # Calculate trends
        if len(timestamps) > 1:
            time_diffs = [(t - timestamps[0]).total_seconds() for t in timestamps]
            reward_trend = np.polyfit(time_diffs, rewards, 1)[0] if len(rewards) > 1 else 0
            response_trend = np.polyfit(time_diffs, response_times, 1)[0] if len(response_times) > 1 else 0
        else:
            reward_trend = response_trend = 0

        return {
            'reward_trend': float(reward_trend),
            'response_time_trend': float(response_trend),
            'duration': (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0
        }

    def _analyze_temporal_patterns(self, current_time: datetime, interval: str) -> Dict[str, Any]:
        """Analyze behavior patterns within specific time intervals."""
        if not self.behavior_history:
            return {}

        # Determine time window
        if interval == 'hour':
            window = timedelta(hours=1)
            num_windows = 24
        elif interval == 'day':
            window = timedelta(days=1)
            num_windows = 7
        else:  # week
            window = timedelta(weeks=1)
            num_windows = 4

        # Initialize statistics data
        stats = defaultdict(lambda: {
            'count': 0,
            'rewards': [],
            'response_times': [],
            'action_types': defaultdict(int)
        })

        # Collect data
        for entry in self.behavior_history:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            window_key = (current_time - timestamp) // window
            if 0 <= window_key < num_windows:
                data = entry['data']
                stats[int(window_key)]['count'] += 1
                if 'reward' in data:
                    stats[int(window_key)]['rewards'].append(data['reward'])
                if 'response_time' in data:
                    stats[int(window_key)]['response_times'].append(data['response_time'])
                if 'action_type' in data:
                    stats[int(window_key)]['action_types'][data['action_type']] += 1

        # Process statistical results
        result = {}
        for window_idx, data in stats.items():
            result[str(window_idx)] = {
                'behavior_count': data['count'],
                'avg_reward': float(np.mean(data['rewards'])) if data['rewards'] else 0,
                'avg_response_time': float(np.mean(data['response_times'])) if data['response_times'] else 0,
                'action_distribution': dict(data['action_types'])
            }

        return result

    def _analyze_cluster_patterns(self, clusters: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Analyze behavior patterns in clusters."""
        patterns = {}

        # Basic statistics
        for cluster_id, cluster_data in clusters.items():
            # Extract cluster behavior data
            actions = [d['action'] for d in cluster_data]
            rewards = [d['reward'] for d in cluster_data]

            # Calculate cluster behavior characteristics
            pattern_info = {
                'size': len(cluster_data),
                'variance': float(np.var(rewards)),
                'mean_reward': float(np.mean(rewards))
            }

            # Analyze behavior features
            patterns[cluster_id] = pattern_info

        return patterns

    def _analyze_temporal_trends(self, cluster_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal trends in behavior cluster."""
        if not cluster_data:
            return {}

        # Extract time series data
        timestamps = [datetime.fromisoformat(d['timestamp']) for d in cluster_data]
        rewards = [d['reward'] for d in cluster_data]

        # Calculate trends
        trend_data = {
            'period_start': timestamps[0].isoformat(),
            'period_end': timestamps[-1].isoformat(),
            'reward_trend': float(np.polyfit(range(len(rewards)), rewards, 1)[0])
        }

        return trend_data

    def _analyze_time_window_patterns(self, window_size: int = 3600) -> Dict[str, Any]:
        """Analyze behavior patterns in specific time intervals."""
        if not self.behavior_history:
            return {}

        # Determine time window
        current_time = datetime.now()
        window_start = current_time - timedelta(seconds=window_size)

        # Initialize statistics
        window_stats = {
            'action_counts': {},
            'reward_stats': {
                'mean': 0.0,
                'std': 0.0
            }
        }

        # Collect data
        window_data = [
            entry for entry in self.behavior_history
            if datetime.fromisoformat(entry['timestamp']) >= window_start
        ]

        if not window_data:
            return window_stats

        # Process statistics
        actions = [d['action'] for d in window_data]
        rewards = [d['reward'] for d in window_data]

        window_stats['action_counts'] = dict(Counter(actions))
        window_stats['reward_stats'] = {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards))
        }

        return window_stats

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not self.behavior_history:
            return {}

        # Extract recent behavior data
        recent_data = self.behavior_history[-100:]
        rewards = [d['reward'] for d in recent_data]
        response_times = [d['response_time'] for d in recent_data]

        # Calculate performance metrics
        metrics = {
            'mean_reward': float(np.mean(rewards)),
            'reward_stability': float(1.0 / (1.0 + np.std(rewards))),
            'mean_response_time': float(np.mean(response_times)),
            'response_time_stability': float(1.0 / (1.0 + np.std(response_times)))
        }

        # Calculate long-term trends
        if len(self.behavior_history) > 100:
            long_term_rewards = [d['reward'] for d in self.behavior_history]
            metrics['long_term_trend'] = float(np.polyfit(range(len(long_term_rewards)), long_term_rewards, 1)[0])

        return metrics

    def _calculate_prediction_accuracy_trend(self) -> Dict[str, float]:
        """Calculate recent prediction accuracy trend."""
        if not self.prediction_history:
            return {}

        # Get recent prediction results
        recent_predictions = self.prediction_history[-50:]
        accuracies = []

        for pred in recent_predictions:
            predicted_state = pred['predicted_state']
            next_state = pred['next_state']

            # Simple state comparison (can be enhanced based on state representation)
            actual = hash(str(next_state)) % 3  # Consistent with _create_prediction_labels
            predicted = hash(str(predicted_state)) % 3

            accuracies.append(1.0 if actual == predicted else 0.0)

        # Calculate accuracy metrics
        if accuracies:
            return {
                'recent_accuracy': float(np.mean(accuracies)),
                'accuracy_trend': float(np.polyfit(range(len(accuracies)), accuracies, 1)[0])
            }

        return {}

    def _evaluate_prediction_stability(self) -> Dict[str, float]:
        """Evaluate prediction stability."""
        if len(self.prediction_history) < 2:
            return {}

        # Collect consecutive prediction changes
        prediction_changes = []
        for i in range(1, len(self.prediction_history)):
            prev_pred = self.prediction_history[i-1]['predicted_state']
            curr_pred = self.prediction_history[i]['predicted_state']

            # Calculate prediction change magnitude
            change = abs(hash(str(curr_pred)) - hash(str(prev_pred))) % 100
            prediction_changes.append(change)

        if prediction_changes:
            return {
                'stability_score': float(1.0 / (1.0 + np.mean(prediction_changes))),
                'change_trend': float(np.polyfit(range(len(prediction_changes)), prediction_changes, 1)[0])
            }

        return {}

    def _analyze_pattern_evolution(self) -> Dict[str, Any]:
        """Analyze behavior pattern evolution."""
        if len(self.behavior_history) < 10:
            return {}

        # Analyze pattern distribution by time windows
        window_size = max(len(self.behavior_history) // 10, 1)  # Split history into 10 windows
        windows = [self.behavior_history[i:i+window_size] 
                  for i in range(0, len(self.behavior_history), window_size)]

        evolution_data = []
        for window in windows:
            if not window:
                continue

            window_patterns = self._identify_patterns(window)
            evolution_data.append({
                'timestamp': window[-1]['timestamp'],
                'patterns': window_patterns
            })

        return {
            'evolution_data': evolution_data,
            'window_size': window_size
        }

    def generate_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generate behavior optimization recommendations."""
        recommendations = []

        # Get recent prediction accuracy trend
        accuracy_trend = self._calculate_prediction_accuracy_trend()

        # Analyze prediction model feature importance
        if self.prediction_model and hasattr(self.prediction_model, 'feature_importances_'):
            feature_importance = self.prediction_model.feature_importances_
            if min(feature_importance) < 0.1:
                recommendations.append({
                    'category': 'model_features',
                    'severity': 'medium',
                    'message': 'Some features have low importance, consider feature selection or engineering'
                })

        # Analyze pattern distribution
        patterns = self._identify_patterns(self.behavior_history)
        if patterns:
            pattern_counts = Counter(patterns)
            max_pattern_ratio = max(pattern_counts.values()) / len(patterns)
            if max_pattern_ratio > 0.7:
                recommendations.append({
                    'category': 'behavior_diversity',
                    'severity': 'high',
                    'message': 'Agent behavior patterns too uniform, recommend increasing behavior diversity'
                })

        # Analyze prediction confidence
        recent_predictions = self.prediction_history[-20:] if self.prediction_history else []
        if recent_predictions:
            avg_confidence = np.mean([p['confidence'] for p in recent_predictions])
            if avg_confidence < 0.6:
                recommendations.append({
                    'category': 'prediction_accuracy',
                    'severity': 'high',
                    'message': 'Low behavior prediction accuracy, recommend collecting more training data or optimizing model'
                })

        # Analyze behavior efficiency
        if self.behavior_history:
            response_times = [b['response_time'] for b in self.behavior_history[-50:]]
            if np.mean(response_times) > 1.0:
                recommendations.append({
                    'category': 'efficiency',
                    'severity': 'medium',
                    'message': 'Long response times, recommend optimizing decision process'
                })

        # Analyze reward trend
        if len(self.behavior_history) > 50:
            rewards = [b['reward'] for b in self.behavior_history[-50:]]
            reward_trend = np.polyfit(range(len(rewards)), rewards, 1)[0]
            if reward_trend < 0:
                recommendations.append({
                    'category': 'learning_effectiveness',
                    'severity': 'high',
                    'message': 'Declining learning effectiveness, recommend adjusting learning strategy or retraining model'
                })

        return recommendations