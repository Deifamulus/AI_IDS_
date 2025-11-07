from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
from collections import defaultdict, deque
import threading
import time
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class Alert:
    timestamp: str
    src_ip: str
    src_port: int
    dest_ip: str
    dest_port: int
    protocol: str
    signature: str
    severity: int
    category: str

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-123'
socketio = SocketIO(app, async_mode='threading')

# Data storage
traffic_stats = {
    'total_packets': 0,
    'protocols': defaultdict(int),
    'top_sources': defaultdict(int),
    'top_destinations': defaultdict(int),
    'packet_sizes': [],
    'recent_packets': deque(maxlen=100),
    'alerts': deque(maxlen=100),  # Store recent alerts
    'alert_count': 0,  # Total alert count
    'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

alert_lock = threading.Lock()  # For thread-safe alert operations

def update_dashboard(packet_info):
    """Update dashboard data with new packet information"""
    traffic_stats['total_packets'] += 1
    traffic_stats['protocols'][packet_info['protocol']] += 1
    
    src = f"{packet_info['src_ip']}:{packet_info['src_port']}"
    dst = f"{packet_info['dst_ip']}:{packet_info['dst_port']}"
    
    traffic_stats['top_sources'][src] += 1
    traffic_stats['top_destinations'][dst] += 1
    traffic_stats['packet_sizes'].append(packet_info['length'])
    
    # Check if this is an alert
    if packet_info.get('is_alert', False):
        with alert_lock:
            alert = Alert(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                src_ip=packet_info['src_ip'],
                src_port=packet_info['src_port'],
                dest_ip=packet_info['dst_ip'],
                dest_port=packet_info['dst_port'],
                protocol=packet_info['protocol'],
                signature=packet_info.get('signature', 'Unknown'),
                severity=packet_info.get('severity', 3),
                category=packet_info.get('category', 'unknown')
            )
            traffic_stats['alerts'].appendleft(asdict(alert))
            traffic_stats['alert_count'] += 1
            
            # Emit alert event
            socketio.emit('new_alert', asdict(alert))
    
    traffic_stats['recent_packets'].appendleft({
        'time': datetime.now().strftime("%H:%M:%S.%f")[:-3],
        'src': src,
        'dst': dst,
        'protocol': packet_info['protocol'],
        'length': packet_info['length'],
        'is_alert': packet_info.get('is_alert', False)
    })
    
    # Keep only last 1000 packet sizes for the chart
    if len(traffic_stats['packet_sizes']) > 1000:
        traffic_stats['packet_sizes'] = traffic_stats['packet_sizes'][-1000:]
    
    # Emit update to all connected clients
    socketio.emit('update', {
        'total_packets': traffic_stats['total_packets'],
        'protocols': dict(traffic_stats['protocols']),
        'top_sources': dict(sorted(traffic_stats['top_sources'].items(), 
                                 key=lambda x: x[1], reverse=True)[:5]),
        'top_destinations': dict(sorted(traffic_stats['top_destinations'].items(),
                                      key=lambda x: x[1], reverse=True)[:5]),
        'recent_packets': list(traffic_stats['recent_packets'])[:20],
        'packet_sizes': traffic_stats['packet_sizes'][-50:],
        'uptime': str(datetime.now() - datetime.strptime(traffic_stats['start_time'], 
                                                       "%Y-%m-%d %H:%M:%S")),
        'alert_count': traffic_stats['alert_count'],
        'recent_alerts': list(traffic_stats['alerts'])[:10]  # Send recent alerts
    })
    
@app.route('/alerts')
def get_alerts():
    """Get recent alerts"""
    return jsonify({
        'alerts': list(traffic_stats['alerts']), 
        'total': traffic_stats['alert_count']
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update', methods=['POST'])
def update():
    """Handle incoming packet data from the NIDS"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data received'}), 400
            
        # Update dashboard with new packet data
        update_dashboard(data)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
