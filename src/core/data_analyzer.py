from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class DataAnalyzer:
    """Analyzer for system performance metrics."""

    def __init__(self, history_window: int = 3600, prediction_horizon: int = 300):
        """Initialize the data analyzer.

        Args:
            history_window: Time window in seconds for historical data analysis.
            prediction_horizon: Time horizon in seconds for future predictions.
        """
        self.history_window = history_window
        self.prediction_horizon = prediction_horizon
        self._metrics_history: List[Dict[str, Any]] = []
        self._models: Dict[str, RandomForestRegressor] = {}
        self._scalers: Dict[str, StandardScaler] = {}

    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add new metrics data point to history.

        Args:
            metrics: System metrics data point.
        """
        self._metrics_history.append({
            'timestamp': datetime.now(),
            'data': metrics
        })
        self._cleanup_old_data()

    def _cleanup_old_data(self) -> None:
        """Remove data points older than history window."""
        cutoff_time = datetime.now() - timedelta(seconds=self.history_window)
        self._metrics_history = [
            item for item in self._metrics_history
            if item['timestamp'] >= cutoff_time
        ]

    def get_resource_trends(self) -> Dict[str, Any]:
        """Calculate resource usage trends.

        Returns:
            Dictionary containing trend analysis for different metrics.
        """
        if not self._metrics_history:
            return {}

        cpu_trend = self._calculate_trend('cpu')
        memory_trend = self._calculate_trend('memory')
        disk_trend = self._calculate_trend('disk')
        gpu_trend = self._calculate_trend('gpu') if self._has_gpu_data() else None

        return {
            'cpu': cpu_trend,
            'memory': memory_trend,
            'disk': disk_trend,
            'gpu': gpu_trend,
            'analysis_time': datetime.now().isoformat()
        }

    def _calculate_trend(self, metric_type: str) -> Dict[str, Any]:
        """Calculate trend for a specific metric type.

        Args:
            metric_type: Type of metric to analyze.

        Returns:
            Dictionary containing trend analysis.
        """
        values = self._extract_metric_values(metric_type)
        if not values:
            return {}

        values_array = np.array(values)
        return {
            'current': float(values_array[-1]),
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'trend': self._calculate_trend_direction(values_array)
        }

    def _extract_metric_values(self, metric_type: str) -> List[float]:
        """Extract values for a specific metric type from history.

        Args:
            metric_type: Type of metric to extract.

        Returns:
            List of metric values.
        """
        values = []
        for item in self._metrics_history:
            if metric_type == 'cpu':
                values.append(np.mean(item['data']['cpu']['percent']))
            elif metric_type == 'memory':
                values.append(item['data']['memory']['virtual']['percent'])
            elif metric_type == 'disk':
                values.append(item['data']['disk']['usage']['percent'])
            elif metric_type == 'gpu' and 'gpu' in item['data']:
                util = item['data']['gpu'].get('utilization')
                if util is not None:
                    values.append(util)
                else:
                    mem = item['data']['gpu']['memory']
                    values.append((mem['allocated'] / mem['reserved']) * 100)
        return values

    def _calculate_trend_direction(self, values: np.ndarray) -> str:
        """Calculate trend direction based on recent values.

        Args:
            values: Array of metric values.

        Returns:
            Trend direction ('increasing', 'decreasing', or 'stable').
        """
        if len(values) < 2:
            return 'stable'

        # 使用最近的值计算趋势
        recent_values = values[-10:] if len(values) > 10 else values
        slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]

        if abs(slope) < 0.1:  # 阈值可调整
            return 'stable'
        return 'increasing' if slope > 0 else 'decreasing'

    def _has_gpu_data(self) -> bool:
        """Check if GPU data is available in history.

        Returns:
            Boolean indicating if GPU data exists.
        """
        return any('gpu' in item['data'] for item in self._metrics_history)

    def get_anomalies(self) -> Dict[str, List[Dict[str, Any]]]:
        """Detect anomalies in system metrics using advanced statistical methods.

        Returns:
            Dictionary containing detected anomalies for each metric type.
        """
        anomalies = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'gpu': []
        }

        for metric_type in ['cpu', 'memory', 'disk', 'gpu']:
            if metric_type == 'gpu' and not self._has_gpu_data():
                continue

            values = self._extract_metric_values(metric_type)
            if not values:
                continue

            values_array = np.array(values)
            # 使用指数加权移动平均和标准差
            ewm_mean = pd.Series(values_array).ewm(span=20).mean()
            ewm_std = pd.Series(values_array).ewm(span=20).std()
            threshold = 3 * ewm_std  # 使用动态阈值

            for i, (value, mean, std) in enumerate(zip(values_array, ewm_mean, ewm_std)):
                if abs(value - mean) > std:
                    severity = 'critical' if abs(value - mean) > 2 * std else 'warning'
                    anomalies[metric_type].append({
                        'timestamp': self._metrics_history[i]['timestamp'].isoformat(),
                        'value': float(value),
                        'deviation': float(abs(value - mean)),
                        'severity': severity,
                        'threshold': float(std)
                    })

        return anomalies

    def train_prediction_models(self) -> Dict[str, Dict[str, float]]:
        """Train machine learning models for resource usage prediction.

        Returns:
            Dictionary containing model performance metrics.
        """
        performance_metrics = {}
        for metric_type in ['cpu', 'memory', 'disk', 'gpu']:
            if metric_type == 'gpu' and not self._has_gpu_data():
                continue

            values = self._extract_metric_values(metric_type)
            if len(values) < 100:  # 需要足够的历史数据
                continue

            X, y = self._prepare_training_data(values)
            if len(X) < 2:
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # 标准化数据
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 训练模型
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test_scaled)
            performance = {
                'mse': float(mean_squared_error(y_test, y_pred)),
                'r2': float(r2_score(y_test, y_pred))
            }
            
            # 保存模型和scaler
            self._models[metric_type] = model
            self._scalers[metric_type] = scaler
            performance_metrics[metric_type] = performance

        return performance_metrics

    def predict_resource_usage(self, metric_type: str, horizon_steps: int = 10) -> Optional[List[float]]:
        """预测未来资源使用情况。

        Args:
            metric_type: 要预测的指标类型
            horizon_steps: 预测步数

        Returns:
            预测值列表，如果无法预测则返回None
        """
        if metric_type not in self._models or metric_type not in self._scalers:
            return None

        values = self._extract_metric_values(metric_type)
        if len(values) < 10:
            return None

        # 准备预测数据
        X = self._prepare_prediction_data(values)
        X_scaled = self._scalers[metric_type].transform(X)
        
        # 进行预测
        predictions = []
        current_input = X_scaled[-1:]
        
        for _ in range(horizon_steps):
            pred = self._models[metric_type].predict(current_input)[0]
            predictions.append(pred)
            
            # 更新输入数据用于下一步预测
            current_input = np.roll(current_input, -1)
            current_input[0, -1] = pred

        return predictions

    def _prepare_training_data(self, values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """准备模型训练数据。

        Args:
            values: 历史数据值列表

        Returns:
            特征矩阵和目标值数组
        """
        window_size = 10
        X, y = [], []
        
        for i in range(len(values) - window_size):
            X.append(values[i:i + window_size])
            y.append(values[i + window_size])
        
        return np.array(X), np.array(y)

    def _prepare_prediction_data(self, values: List[float]) -> np.ndarray:
        """准备预测输入数据。

        Args:
            values: 历史数据值列表

        Returns:
            预处理后的输入特征矩阵
        """
        window_size = 10
        if len(values) < window_size:
            return np.array([])
        
        return np.array([values[-window_size:]])