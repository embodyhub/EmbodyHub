from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
import json
import csv
import io

from ..core.behavior_analyzer import BehaviorAnalyzer
from ..core.advanced_report_generator import AdvancedReportGenerator

router = APIRouter()
behavior_analyzer = BehaviorAnalyzer()
report_generator = AdvancedReportGenerator()

@router.get("/behavior/analysis")
async def get_behavior_analysis() -> Dict[str, Any]:
    """Get behavior analysis data."""
    analysis_report = behavior_analyzer.generate_analysis_report()
    
    # Format data to meet frontend display requirements
    response_data = {
        'totalBehaviors': analysis_report['total_behaviors'],
        'performance': {
            'avgReward': analysis_report['performance_metrics']['recent_performance']['avg_reward'],
            'avgResponseTime': analysis_report['performance_metrics']['recent_performance']['avg_response_time']
        },
        'predictions': {
            'accuracy': analysis_report.get('predictions', {}).get('recent_performance', {}).get('accuracy_trend', {}).get('recent_accuracy', 0)
        },
        'patterns': [
            {
                'name': name,
                'value': data['size']
            } for name, data in analysis_report.get('patterns', {}).items()
        ],
        'rewardTrend': {
            'timestamps': [],
            'values': []
        },
        'timeline': {
            'hour': _format_temporal_data(analysis_report['trends']['hourly']),
            'day': _format_temporal_data(analysis_report['trends']['daily']),
            'week': _format_temporal_data(analysis_report['trends']['weekly'])
        }
    }
    
    return response_data

@router.get("/behavior/report")
async def export_behavior_report(format: str = 'json'):
    """Export behavior analysis report."""
    analysis_report = behavior_analyzer.generate_analysis_report()
    
    if format == 'json':
        return StreamingResponse(
            io.StringIO(json.dumps(analysis_report, indent=2)),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=behavior_report.json"}
        )
    elif format == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        _write_report_to_csv(writer, analysis_report)
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=behavior_report.csv"}
        )
    elif format == 'html':
        html_content = report_generator.generate_advanced_behavior_report(analysis_report, 'html')
        return Response(
            content=html_content,
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename=behavior_report.html"}
        )
    else:
        raise ValueError(f"Unsupported format: {format}")

def _format_temporal_data(temporal_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format time series data to adapt to frontend chart display."""
    timestamps = sorted(temporal_data.keys())
    explore_data = []
    exploit_data = []
    learn_data = []
    
    for timestamp in timestamps:
        data = temporal_data[timestamp]
        action_dist = data.get('action_distribution', {})
        explore_data.append(action_dist.get('explore', 0))
        exploit_data.append(action_dist.get('exploit', 0))
        learn_data.append(action_dist.get('learn', 0))
    
    return {
        'timestamps': timestamps,
        'explore': explore_data,
        'exploit': exploit_data,
        'learn': learn_data
    }

def _write_report_to_csv(writer: csv.writer, report: Dict[str, Any]) -> None:
    """Write report data to CSV format."""
    # Write basic information
    writer.writerow(['Report Generated', report['timestamp']])
    writer.writerow(['Total Behaviors', report['total_behaviors']])
    writer.writerow([])
    
    # Write performance metrics
    writer.writerow(['Performance Metrics'])
    metrics = report['performance_metrics']['recent_performance']
    for key, value in metrics.items():
        writer.writerow([key, value])
    writer.writerow([])
    
    # Write behavior patterns
    writer.writerow(['Behavior Patterns'])
    for pattern, data in report.get('patterns', {}).items():
        writer.writerow([pattern, data['size'], data['percentage']])
    writer.writerow([])
    
    # Write prediction performance
    if 'predictions' in report:
        writer.writerow(['Prediction Performance'])
        pred_metrics = report['predictions'].get('recent_performance', {})
        for key, value in pred_metrics.items():
            writer.writerow([key, value])