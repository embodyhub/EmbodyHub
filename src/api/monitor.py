from fastapi import APIRouter, WebSocket
from typing import Dict, Any
from core.system_monitor import SystemMonitor
import asyncio
import json

router = APIRouter()
system_monitor = SystemMonitor()

@router.get("/system-metrics")
async def get_system_metrics() -> Dict[str, Any]:
    """Get current system resource usage metrics."""
    return system_monitor.get_latest_metrics()

@router.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time system metrics."""
    await websocket.accept()
    try:
        while True:
            metrics = system_monitor.get_latest_metrics()
            await websocket.send_text(json.dumps(metrics))
            await asyncio.sleep(1)  # Update interval
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()