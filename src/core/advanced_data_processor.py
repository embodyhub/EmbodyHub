\"""Advanced Data Processing Module

This module provides enhanced data processing capabilities for
advanced behavior analysis and prediction in EmbodyHub.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

class AdvancedDataProcessor:
    """Advanced data processor providing enhanced data analysis and feature extraction capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced data processor.

        Args:
            config: Configuration parameter dictionary
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.anomaly_detector = IsolationForest(random_state=42)
        self.feature_importance = {}

    def process_behavior_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process behavior data and extract advanced features.

        Args:
            data: List of raw behavior data

        Returns:
            Processed features and analysis results
        """
        if not data:
            return {}

        # Extract base features
        features = self._extract_base_features(data)
        if not isinstance(features, np.ndarray) or len(features) == 0:
            return {}

        # Data standardization
        scaled_features = self.scaler.fit_transform(features)

        # Dimensionality reduction analysis
        reduced_features = self.pca.fit_transform(scaled_features)

        # Exception detection
        anomalies = self._detect_anomalies(scaled_features)

        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(features)

        return {
            'processed_features': reduced_features.tolist(),
            'feature_importance': feature_importance,
            'anomalies': anomalies,
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'processing_timestamp': datetime.now().isoformat()
        }

    def analyze_temporal_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze time series periodicity."""
        if not data:
            return {}

        # Extract time features
        temporal_features = self._extract_temporal_features(data)
        if not temporal_features:
            return {}

        # Analyze periodicity
        periodicity = self._analyze_periodicity(temporal_features)

        # Detect trend change points
        change_points = self._detect_change_points(temporal_features)

        return {
            'periodicity': periodicity,
            'change_points': change_points,
            'temporal_patterns': self._identify_temporal_patterns(temporal_features)
        }

    def _extract_base_features(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Extract base features."""

        # Basic numerical features
        features = []
        for d in data:
            feature_vector = [
                d.get('action_value', 0),
                d.get('confidence', 0),
                d.get('response_time', 0),
                d.get('reward', 0)
            ]

            # Action type encoding
            action_type = d.get('action_type', 'unknown')
            action_encoding = [1 if t == action_type else 0 for t in self.action_types]
            feature_vector.extend(action_encoding)

            # Time features
            dt = datetime.fromisoformat(d['timestamp'])
            time_features = [
                float(dt.hour) / 24.0,  # Hour normalized
                float(dt.weekday()) / 7.0,  # Weekday normalized
                np.sin(2 * np.pi * dt.hour / 24),  # Periodic time features
                np.cos(2 * np.pi * dt.hour / 24)
            ]
            feature_vector.extend(time_features)
            features.append(feature_vector)

        return np.array(features)

    def _extract_time_series_features(self, data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract time series features."""

        # Extract base time series data
        timestamps = [datetime.fromisoformat(d['timestamp']) for d in data]
        values = {
            'action_value': [d.get('action_value', 0) for d in data],
            'confidence': [d.get('confidence', 0) for d in data],
            'reward': [d.get('reward', 0) for d in data]
        }

        # Calculate time differences and rates of change
        time_diffs = np.diff([t.timestamp() for t in timestamps])
        features = {}
        for key, val in values.items():
            features[f'{key}_diff'] = np.diff(val)
            features[f'{key}_rate'] = features[f'{key}_diff'] / time_diffs

        # Calculate rolling statistics
        windows = [3, 5, 10]
        for w in windows:
            for key, val in values.items():
                val_array = np.array(val)
                features[f'{key}_mean_{w}'] = np.convolve(val_array, np.ones(w)/w, mode='valid')
                features[f'{key}_std_{w}'] = np.array([np.std(val_array[max(0, i-w):i+1]) 
                                                      for i in range(len(val_array))])

        return features

    def _detect_anomalies(self, features: np.ndarray) -> np.ndarray:
        """Detect anomalous behaviors."""
        return self.anomaly_detector.predict(features) if self.anomaly_detector else np.zeros(len(features))

    def _compute_feature_importance(self, features: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculate feature importance."""
        if not self.feature_selector:
            return np.ones(features.shape[1])

        # Use variance as feature importance indicator
        return np.var(features, axis=0)

    def _analyze_periodicity(self, values: np.ndarray) -> Dict[str, float]:
        """Analyze time series periodicity."""
        if len(values) < 2:
            return {}

        # Use autocorrelation analysis to detect periodicity
        n = len(values)
        autocorr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')[n-1:]
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find significant peaks
        peaks = self._find_peaks(autocorr)
        significant_peaks = [p for p in peaks if autocorr[p] > 0.3]  # Set significance threshold

        # Analyze period characteristics
        if significant_peaks:
            periods = np.diff(significant_peaks)
            main_period = float(np.median(periods))  # Use median as main period
            period_std = float(np.std(periods))  # Period stability
        else:
            main_period = 0
            period_std = 0

        # Calculate period strength and consistency
        return {
            'main_period': main_period,
            'period_stability': 1.0 / (1.0 + period_std) if period_std > 0 else 1.0,
            'strength': float(np.max(autocorr[1:])) if len(autocorr) > 1 else 0
        }

    def _detect_change_points(self, temporal_features: Dict[str, np.ndarray]) -> Dict[str, List[int]]:
        """Detect change points in time series data."""
        change_points = {}
        window_size = 5

        for metric, values in temporal_features['values'].items():
            if len(values) < window_size * 2:
                continue

            try:
                # Calculate changes at multiple time scales
                changes = {}
                for w in [3, 5, 7]:  # Multiple window sizes
                    means = np.array([np.mean(values[i:i+w]) 
                                    for i in range(len(values)-w+1)])
                    changes[w] = np.abs(np.diff(means))

                # Combine change points from multiple scales
                combined_changes = np.zeros(len(values)-1)
                for w, change in changes.items():
                    # Standardize and combine changes from different scales
                    if len(change) > 0:
                        std_change = (change - np.mean(change)) / (np.std(change) + 1e-6)
                        combined_changes[:len(std_change)] += std_change

                # Use dynamic thresholds to detect change points
                rolling_std = np.array([np.std(combined_changes[max(0, i-window_size):i+1])
                                      for i in range(len(combined_changes))])
                thresholds = rolling_std * 2 + np.mean(combined_changes)

                # Find significant change points
                change_points[metric] = [i for i, (change, threshold) 
                                       in enumerate(zip(combined_changes, thresholds))
                                       if change > threshold]

                # Calculate change point characteristics
                if change_points[metric]:
                    changes_info = []
                    for cp in change_points[metric]:
                        if cp > 0 and cp < len(values) - 1:
                            before = np.mean(values[max(0, cp-window_size):cp])
                            after = np.mean(values[cp:min(len(values), cp+window_size)])
                            change_magnitude = abs(after - before) / (abs(before) + 1e-6)
                            changes_info.append({
                                'position': cp,
                                'magnitude': float(change_magnitude),
                                'direction': 1 if after > before else -1
                            })
                    change_points[metric] = changes_info

            except Exception as e:
                print(f"Error detecting change points for {metric}: {e}")
                change_points[metric] = []

        return change_points

    def _identify_temporal_patterns(self, temporal_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Identify patterns in time series data."""
        patterns = {}
        for metric, values in temporal_features['values'].items():
            if len(values) < 3:
                continue

            # Calculate trend
            trend = np.polyfit(range(len(values)), values, 1)[0]
            
            # Calculate volatility
            volatility = np.std(values)/np.mean(values) if np.mean(values) != 0 else 0

            patterns[metric] = {
                'trend': float(trend),
                'volatility': float(volatility),
                'range': {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values))
                }
            }

        return patterns

    def _find_peaks(self, arr: np.ndarray) -> List[int]:
        """Find local maxima positions in an array."""
        peaks = []
        for i in range(1, len(arr)-1):
            if arr[i-1] < arr[i] and arr[i] > arr[i+1]:
                peaks.append(i)
        return peaks