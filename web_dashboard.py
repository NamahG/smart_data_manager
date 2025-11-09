#!/usr/bin/env python3
"""
Intelligent Data Fabric - Interactive Web Dashboard
Real-time visualization with Flask + SocketIO
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import json
import os
from pathlib import Path
from datetime import datetime
import threading
import time

app = Flask(__name__,
            template_folder='web/templates',
            static_folder='web/static')
app.config['SECRET_KEY'] = 'intelligent-data-fabric-2025'
socketio = SocketIO(app, cors_allowed_origins="*")

BASE_DIR = Path(__file__).parent
EXPORT_DIR = BASE_DIR / 'data' / 'exports'

# Global state
dashboard_state = {
    'last_update': None,
    'pipeline_status': 'idle',
    'metrics': {}
}


def load_export_data(filename):
    """Load data from export directory"""
    file_path = EXPORT_DIR / filename
    if file_path.exists():
        with open(file_path, 'r') as f:
            return json.load(f)
    return None


def watch_exports():
    """Watch export directory for changes and push updates"""
    last_mod_times = {}

    while True:
        try:
            for export_file in EXPORT_DIR.glob('*.json'):
                mod_time = export_file.stat().st_mtime

                if export_file.name not in last_mod_times or last_mod_times[export_file.name] < mod_time:
                    last_mod_times[export_file.name] = mod_time

                    # File was updated, push to clients
                    data = load_export_data(export_file.name)
                    if data:
                        socketio.emit('data_update', {
                            'file': export_file.name,
                            'data': data,
                            'timestamp': datetime.now().isoformat()
                        }, namespace='/')

            time.sleep(2)  # Check every 2 seconds
        except Exception as e:
            print(f"Watch error: {e}")
            time.sleep(5)


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard_interactive.html')


@app.route('/api/status')
def get_status():
    """Get current system status"""
    # Load all export data
    data = {
        'timestamp': datetime.now().isoformat(),
        'optimization': load_export_data('task1_complete_report.json'),
        'migration': load_export_data('migration_report.json'),
        'streaming': load_export_data('streaming_data.json'),
        'ml_predictions': load_export_data('ml_predictions.json'),
        'consistency': load_export_data('consistency_status.json'),
        'security': load_export_data('security_policies.json'),
        'alerts': load_export_data('system_alerts.json'),
        'dashboard': load_export_data('dashboard_data.json')
    }

    return jsonify(data)


@app.route('/api/metrics')
def get_metrics():
    """Get aggregated metrics"""
    dashboard_data = load_export_data('dashboard_data.json')

    if not dashboard_data:
        return jsonify({'error': 'No data available. Run the pipeline first.'})

    return jsonify(dashboard_data)


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f'Client connected: {datetime.now()}')
    emit('connected', {'message': 'Connected to Intelligent Data Fabric'})

    # Send current state
    status_data = get_status().get_json()
    emit('initial_data', status_data)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f'Client disconnected: {datetime.now()}')


@socketio.on('request_update')
def handle_update_request():
    """Handle manual update request"""
    status_data = get_status().get_json()
    emit('data_update', status_data)


def find_free_port(start_port=5001, max_attempts=10):
    """Find a free port starting from start_port"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return start_port  # Fallback

if __name__ == '__main__':
    # Find free port (avoid 5000 which is used by AirPlay on macOS)
    port = find_free_port(5001)

    print("\n" + "="*80)
    print("  INTELLIGENT DATA FABRIC - INTERACTIVE DASHBOARD")
    print("="*80)
    print(f"\n  Dashboard URL: http://localhost:{port}")
    print(f"  Data Directory: {EXPORT_DIR}")
    print(f"\n  Features:")
    print(f"    - Real-time updates via WebSocket")
    print(f"    - Interactive charts and metrics")
    print(f"    - Auto-refresh when pipeline runs")
    print(f"\n  Press Ctrl+C to stop")
    print("="*80 + "\n")

    # Start export watcher in background
    watcher_thread = threading.Thread(target=watch_exports, daemon=True)
    watcher_thread.start()

    # Run Flask app
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
