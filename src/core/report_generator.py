import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

class ReportGenerator:
    """Unified report generator that supports exporting reports in multiple formats."""

    def __init__(self, base_dir: str = 'reports'):
        """Initialize the report generator.

        Args:
            base_dir: Base directory for saving reports.
        """
        self.base_dir = Path(base_dir)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure the required directory structure exists."""
        for subdir in ['behavior', 'performance', 'combined']:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

    def generate_behavior_report(self, data: Dict[str, Any], format: str = 'json') -> str:
        """Generate behavior analysis report.

        Args:
            data: Behavior analysis data.
            format: Report format ('json', 'csv', 'html').

        Returns:
            Path to the generated report file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f'behavior_report_{timestamp}'
        return self._save_report(data, 'behavior', report_name, format)

    def generate_performance_report(self, data: Dict[str, Any], format: str = 'json') -> str:
        """Generate performance analysis report.

        Args:
            data: Performance analysis data.
            format: Report format ('json', 'csv', 'html').

        Returns:
            Path to the generated report file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f'performance_report_{timestamp}'
        return self._save_report(data, 'performance', report_name, format)

    def generate_combined_report(self, 
                               behavior_data: Dict[str, Any],
                               performance_data: Dict[str, Any],
                               format: str = 'html') -> str:
        """Generate comprehensive analysis report.

        Args:
            behavior_data: Behavior analysis data.
            performance_data: Performance analysis data.
            format: Report format ('json', 'csv', 'html').

        Returns:
            Path to the generated report file.
        """
        combined_data = {
            'timestamp': datetime.now().isoformat(),
            'behavior_analysis': behavior_data,
            'performance_analysis': performance_data,
            'correlations': self._analyze_correlations(behavior_data, performance_data)
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f'combined_report_{timestamp}'
        return self._save_report(combined_data, 'combined', report_name, format)

    def _save_report(self, 
                     data: Dict[str, Any], 
                     report_type: str,
                     report_name: str,
                     format: str) -> str:
        """Save report in the specified format.

        Args:
            data: Report data.
            report_type: Report type ('behavior', 'performance', 'combined').
            report_name: Report name.
            format: Output format.

        Returns:
            Path to the generated report file.
        """
        report_dir = self.base_dir / report_type
        
        if format == 'json':
            file_path = report_dir / f'{report_name}.json'
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif format == 'csv':
            file_path = report_dir / f'{report_name}.csv'
            df = pd.json_normalize(data)
            df.to_csv(file_path, index=False)

        elif format == 'html':
            file_path = report_dir / f'{report_name}.html'
            html_content = self._generate_html_report(data, report_type)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        else:
            raise ValueError(f'Unsupported format: {format}')

        return str(file_path)

    def _generate_html_report(self, data: Dict[str, Any], report_type: str) -> str:
        """Generate HTML format report.

        Args:
            data: Report data.
            report_type: Report type.

        Returns:
            HTML format report content.
        """
        # Base HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_type.capitalize()} Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ margin: 10px 0; }}
                .chart {{ width: 100%; height: 400px; margin: 20px 0; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>{report_type.capitalize()} Analysis Report</h1>
            <div class="timestamp">Generated at: {data['timestamp']}</div>
            {self._generate_html_sections(data, report_type)}
        </body>
        </html>
        """
        return html_template

    def _generate_html_sections(self, data: Dict[str, Any], report_type: str) -> str:
        """Generate content sections of HTML report.

        Args:
            data: Report data.
            report_type: Report type.

        Returns:
            Content sections of HTML report.
        """
        sections = []

        if report_type == 'behavior':
            sections.extend([
                self._generate_patterns_section(data.get('patterns', {})),
                self._generate_predictions_section(data.get('predictions', {})),
                self._generate_trends_section(data.get('trends', {}))
            ])

        elif report_type == 'performance':
            sections.extend([
                self._generate_metrics_section(data.get('response_time', {}), 'Response Time'),
                self._generate_metrics_section(data.get('resource_usage', {}), 'Resource Usage'),
                self._generate_metrics_section(data.get('accuracy', {}), 'Accuracy')
            ])

        elif report_type == 'combined':
            sections.extend([
                self._generate_correlation_section(data.get('correlations', {}))
            ])

        return '\n'.join(sections)

    def _analyze_correlations(self, 
                            behavior_data: Dict[str, Any],
                            performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations between behavior and performance data.

        Args:
            behavior_data: Behavior analysis data.
            performance_data: Performance analysis data.

        Returns:
            Correlation analysis results.
        """
        correlations = {
            'response_time_vs_reward': self._calculate_correlation(
                behavior_data.get('metrics', {}).get('response_times', []),
                performance_data.get('response_time', {}).get('values', [])
            ),
            'accuracy_vs_confidence': self._calculate_correlation(
                performance_data.get('accuracy', {}).get('values', []),
                behavior_data.get('metrics', {}).get('confidences', [])
            ),
            'reward_vs_accuracy': self._calculate_correlation(
                behavior_data.get('metrics', {}).get('rewards', []),
                performance_data.get('accuracy', {}).get('values', [])
            )
        }

        # Add trend analysis
        correlations['trends'] = {
            'reward_trend': self._analyze_trend(behavior_data.get('metrics', {}).get('rewards', [])),
            'accuracy_trend': self._analyze_trend(performance_data.get('accuracy', {}).get('values', []))
        }

        return correlations

    def _analyze_trend(self, values: List[float]) -> Dict[str, float]:
        """Analyze data trends.

        Args:
            values: List of values.

        Returns:
            Trend analysis results.
        """
        if not values or len(values) < 2:
            return {'slope': 0.0, 'variance': 0.0}

        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        variance = np.var(values)

        return {
            'slope': float(slope),
            'variance': float(variance)
        }

    def _create_predictions_chart_script(self, predictions: Dict[str, Any]) -> str:
        """Create JavaScript code for prediction analysis chart."""
        if not predictions or 'recent_performance' not in predictions:
            return ""

        perf = predictions['recent_performance']
        metrics = ['accuracy', 'precision', 'recall']
        values = [perf.get(m, 0) for m in metrics]

        return f"""
        Plotly.newPlot('predictions_chart', [{{
            x: {metrics},
            y: {values},
            type: 'bar',
            name: 'Prediction Performance Metrics'
        }}], {{
            title: 'Prediction Model Performance',
            height: 400,
            yaxis: {{range: [0, 1]}}
        }});
        """

    def _create_trends_chart_script(self, trends: Dict[str, Any]) -> str:
        """Create JavaScript code for trend analysis chart."""
        if not trends:
            return ""

        x_data = []
        y_data = []
        names = []

        for period, data in trends.items():
            if isinstance(data, dict):
                for metric, value in data.items():
                    if isinstance(value, (int, float)):
                        x_data.append(period)
                        y_data.append(value)
                        names.append(metric)

        return f"""
        Plotly.newPlot('trends_chart', [{{
            x: {x_data},
            y: {y_data},
            text: {names},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Trend Analysis'
        }}], {{
            title: 'Performance Trend Analysis',
            height: 400
        }});
        """

    def _create_correlation_chart_script(self, correlations: Dict[str, Any]) -> str:
        """Create JavaScript code for correlation analysis chart."""
        if not correlations:
            return ""

        labels = list(correlations.keys())
        values = [correlations[k] for k in labels if isinstance(correlations[k], (int, float))]

        return f"""
        Plotly.newPlot('correlation_chart', [{{
            x: {labels},
            y: {values},
            type: 'bar',
            name: 'Correlation Analysis'
        }}], {{
            title: 'Metric Correlation Analysis',
            height: 400,
            yaxis: {{range: [-1, 1]}}
        }});
        """
            )
        }
        return correlations

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation between two metrics.

        Args:
            x: Data list for first metric.
            y: Data list for second metric.

        Returns:
            Correlation coefficient.
        """
        if not x or not y or len(x) != len(y):
            return 0.0

        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = (
            (sum((xi - x_mean) ** 2 for xi in x) * 
             sum((yi - y_mean) ** 2 for yi in y)) ** 0.5
        )

        return numerator / denominator if denominator != 0 else 0.0