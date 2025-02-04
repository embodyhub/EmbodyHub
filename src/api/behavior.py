from flask import Blueprint, jsonify, send_file
from datetime import datetime
from ..core.behavior_analyzer import BehaviorAnalyzer
from ..core.advanced_report_generator import AdvancedReportGenerator

behavior_bp = Blueprint('behavior', __name__)
behavior_analyzer = BehaviorAnalyzer()
report_generator = AdvancedReportGenerator()

@behavior_bp.route('/analysis', methods=['GET'])
def get_behavior_analysis():
    """Get behavior analysis data."""
    try:
        # Generate analysis report
        report = behavior_analyzer.generate_analysis_report()
        
        # Extract data required by frontend
        response_data = {
            'patterns': report.get('patterns', {}),
            'performance_metrics': report.get('performance_metrics', {}),
            'predictions': report.get('predictions', {}),
            'recommendations': report.get('recommendations', [])
        }
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@behavior_bp.route('/report', methods=['GET'])
def get_behavior_report():
    """Generate and download behavior analysis report."""
    try:
        # Generate analysis report
        report = behavior_analyzer.generate_analysis_report()
        
        # Generate advanced report
        report_path = report_generator.generate_advanced_behavior_report(
            report,
            format='pdf'
        )
        
        # Send file
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f'behavior_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500