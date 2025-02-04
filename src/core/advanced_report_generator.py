import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from .report_generator import ReportGenerator

class AdvancedReportGenerator(ReportGenerator):
    """Advanced report generator that provides richer data analysis and visualization capabilities."""

    def __init__(self, base_dir: str = 'reports'):
        """Initialize the advanced report generator.

        Args:
            base_dir: Base directory for saving reports.
        """
        super().__init__(base_dir)
        self._setup_advanced_directories()

    def _setup_advanced_directories(self) -> None:
        """Create additional directories required for advanced reports."""
        for subdir in ['visualizations', 'trends', 'predictions']:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

    def generate_advanced_behavior_report(self, data: Dict[str, Any], format: str = 'html') -> str:
        """Generate enhanced behavior analysis report.

        Args:
            data: Behavior analysis data.
            format: Report format ('json', 'csv', 'html', 'pdf').

        Returns:
            Path to the generated report file.
        """
        # Enhance data analysis
        enhanced_data = self._enhance_behavior_data(data)
        
        # Generate visualizations
        visualizations = self._generate_behavior_visualizations(enhanced_data)
        enhanced_data['visualizations'] = visualizations

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f'advanced_behavior_report_{timestamp}'
        return self._save_advanced_report(enhanced_data, 'behavior', report_name, format)

    def generate_advanced_performance_report(self, data: Dict[str, Any], format: str = 'html') -> str:
        """Generate enhanced performance analysis report.

        Args:
            data: Performance analysis data.
            format: Report format ('json', 'csv', 'html', 'pdf').

        Returns:
            Path to the generated report file.
        """
        # Enhance performance data analysis
        enhanced_data = self._enhance_performance_data(data)
        
        # Generate visualizations
        visualizations = self._generate_performance_visualizations(enhanced_data)
        enhanced_data['visualizations'] = visualizations

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f'advanced_performance_report_{timestamp}'
        return self._save_advanced_report(enhanced_data, 'performance', report_name, format)

    def _enhance_behavior_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance behavior data analysis."""
        enhanced_data = data.copy()

        # Add advanced statistical analysis
        if 'patterns' in data:
            enhanced_data['pattern_analysis'] = self._analyze_pattern_evolution(data['patterns'])

        # Add prediction performance evaluation
        if 'predictions' in data:
            enhanced_data['prediction_evaluation'] = self._evaluate_prediction_performance(data['predictions'])

        # Add behavioral trend analysis
        if 'trends' in data:
            enhanced_data['trend_analysis'] = self._analyze_behavioral_trends(data['trends'])

        return enhanced_data

    def _enhance_performance_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance performance data analysis."""
        enhanced_data = data.copy()

        # Add detailed statistics for performance metrics
        metrics = ['response_time', 'resource_usage', 'accuracy']
        for metric in metrics:
            if metric in data:
                enhanced_data[f'{metric}_stats'] = self._calculate_detailed_statistics(data[metric])

        # Add performance trend analysis
        enhanced_data['performance_trends'] = self._analyze_performance_trends(data)

        # Add anomaly detection results
        enhanced_data['anomalies'] = self._detect_performance_anomalies(data)

        return enhanced_data

    def _analyze_pattern_evolution(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the evolution of behavior patterns."""
        return {
            'pattern_stability': self._calculate_pattern_stability(patterns),
            'transition_matrix': self._calculate_pattern_transitions(patterns),
            'dominant_patterns': self._identify_dominant_patterns(patterns)
        }

    def _evaluate_prediction_performance(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate prediction model performance."""
        return {
            'accuracy_trend': self._analyze_prediction_accuracy_trend(predictions),
            'confidence_analysis': self._analyze_prediction_confidence(predictions),
            'error_patterns': self._analyze_prediction_errors(predictions)
        }

    def _analyze_behavioral_trends(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavioral trends."""
        return {
            'seasonal_patterns': self._identify_seasonal_patterns(trends),
            'long_term_trends': self._analyze_long_term_trends(trends),
            'change_points': self._detect_trend_change_points(trends)
        }

    def _generate_behavior_visualizations(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualizations for behavior data."""
        visualizations = {}

        # Add pattern distribution chart
        if 'patterns' in data:
            visualizations['pattern_distribution'] = self._create_pattern_distribution_chart(data['patterns'])

        # Add prediction accuracy trend chart
        if 'predictions' in data:
            visualizations['prediction_accuracy'] = self._create_prediction_accuracy_chart(data['predictions'])

        # Add behavior trend chart
        if 'trends' in data:
            visualizations['behavior_trends'] = self._create_behavior_trends_chart(data['trends'])

        return visualizations

    def _generate_performance_visualizations(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualizations for performance data."""
        visualizations = {}

        # Add performance metrics dashboard
        visualizations['metrics_dashboard'] = self._create_metrics_dashboard(data)

        # Add resource usage trend chart
        if 'resource_usage' in data:
            visualizations['resource_trends'] = self._create_resource_trends_chart(data['resource_usage'])

        # Add performance anomaly detection chart
        if 'anomalies' in data:
            visualizations['anomaly_detection'] = self._create_anomaly_detection_chart(data['anomalies'])

        return visualizations

    def _save_advanced_report(self, 
                            data: Dict[str, Any], 
                            report_type: str,
                            report_name: str,
                            format: str) -> str:
        """Save advanced report.

        Args:
            data: Report data.
            report_type: Report type.
            report_name: Report name.
            format: Output format.

        Returns:
            Path to the generated report file.
        """
        report_dir = self.base_dir / report_type

        if format == 'html':
            file_path = report_dir / f'{report_name}.html'
            html_content = self._generate_advanced_html_report(data, report_type)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        else:
            # Call parent class's save method for other formats
            return super()._save_report(data, report_type, report_name, format)

        return str(file_path)

    def _generate_advanced_html_report(self, data: Dict[str, Any], report_type: str) -> str:
        """Generate advanced HTML report."""
        # Basic HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Advanced {report_type.capitalize()} Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #f8f9fa; }}
                .chart {{ width: 100%; height: 400px; margin: 20px 0; }}
                .header {{ background-color: #007bff; color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; }}
                .summary {{ background-color: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .recommendations {{ background-color: #d4edda; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Advanced {report_type.capitalize()} Analysis Report</h1>
                    <div>Generated at: {data['timestamp']}</div>
                </div>
                {self._generate_advanced_html_sections(data, report_type)}
            </div>
        </body>
        </html>
        """
        return html_template

    def _generate_advanced_html_sections(self, data: Dict[str, Any], report_type: str) -> str:
        """Generate content sections of advanced HTML report."""
        sections = []

        # Add report summary
        sections.append(self._generate_summary_section(data))

        # Add visualization section
        if 'visualizations' in data:
            sections.append(self._generate_visualization_section(data['visualizations']))

        # Add detailed analysis results
        if report_type == 'behavior':
            sections.extend([
                self._generate_pattern_analysis_section(data.get('pattern_analysis', {})),
                self._generate_prediction_evaluation_section(data.get('prediction_evaluation', {})),
                self._generate_trend_analysis_section(data.get('trend_analysis', {}))
            ])
        elif report_type == 'performance':
            sections.extend([
                self._generate_performance_metrics_section(data),
                self._generate_anomaly_section(data.get('anomalies', {}))
            ])

        # Add recommendations and optimization section
        sections.append(self._generate_recommendations_section(data))

        return '\n'.join(sections)