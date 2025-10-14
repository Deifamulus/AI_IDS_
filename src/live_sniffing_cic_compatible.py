"""
Live Network Sniffer Compatible with CIC-Trained Models
This version can use models trained on CIC datasets for real-time detection
"""

import os
import sys
import logging
import argparse
import joblib
import pandas as pd
import numpy as np
from scapy.all import *
from scapy.layers.inet import ETH_P_ALL

# Try to import pcapdnet backend, but make it optional
try:
    import scapy.arch.pcapdnet  # Import pcapdnet backend if available
except (ImportError, ModuleNotFoundError):
    print("[*] Note: pcapdnet backend not available, using default backend")
import csv
import time
import signal
from datetime import datetime
from collections import defaultdict, Counter, deque
from scapy.layers import http as scapy_http
from feature_extractor_patched_final import extract_features, flow_tracker
from typing import Dict, List, Optional, Any, Tuple

# Import enhanced detection components
try:
    from enhanced_detection import (
        AdvancedAttackDetector,
        OptimizedPacketProcessor,
        extract_flow_features,
        calculate_entropy,
        calculate_printable_ratio
    )
    ENHANCED_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"[!] Enhanced detection module not available: {e}")
    print("[*] Falling back to basic detection mode")
    ENHANCED_DETECTION_AVAILABLE = False
    AdvancedAttackDetector = None
    OptimizedPacketProcessor = None
    extract_flow_features = None
    calculate_entropy = None
    calculate_printable_ratio = None

# Configuration
DEFAULT_INTERFACE = "eth0"
LOG_FILENAME = "sniffing_log_cic.txt"
DEBUG_LOG_FILE = "debug_log_cic.csv"
CSV_FILENAME = "captured_features_cic.csv"

# Attack type mapping for CIC-IDS dataset
CIC_ATTACK_TYPES = {
    0: 'Benign',
    1: 'FTP-Patator',
    2: 'SSH-Patator',
    3: 'DoS Hulk',
    4: 'DoS GoldenEye',
    5: 'DoS slowloris',
    6: 'DoS Slowhttptest',
    7: 'Heartbleed',
    8: 'Web Attack - Brute Force',
    9: 'Web Attack - XSS',
    10: 'Web Attack - Sql Injection',
    11: 'Infiltration',
    12: 'Bot',
    13: 'PortScan',
    14: 'DDoS',
    15: 'Brute Force -Web',
    16: 'Brute Force -XSS',
    17: 'SQL Injection',
    18: 'FTP-BruteForce',
    19: 'SSH-BruteForce',
    20: 'DoS attacks-SlowHTTPTest',
    21: 'DoS attacks-Hulk',
    22: 'DoS attacks-Slowloris',
    23: 'DoS attacks-GoldenEye',
    24: 'DDOS attack-LOIC-HTTP',
    25: 'DDOS attack-HOIC',
    26: 'DDOS attack-LOIC-UDP',
    27: 'MSSQL',
    28: 'PostgreSQL',
    29: 'Apache',
    30: 'IIS',
    31: 'Nginx',
    32: 'Lighttpd'
}

def is_benign_traffic(prediction_label, src_port, dst_port, protocol, features=None):
    """
    Check if traffic is likely to be benign based on ports, protocol, and features
    
    Args:
        prediction_label: The predicted attack label
        src_port: Source port
        dst_port: Destination port
        protocol: Network protocol (TCP/UDP/ICMP)
        features: Optional dictionary of extracted features
        
    Returns:
        bool: True if traffic is likely benign, False otherwise
    """
    # Common web ports that are typically safe (HTTP/HTTPS)
    web_ports = {80, 443, 8080, 8443, 8000, 8008, 8081, 8088, 8888, 9000}
    
    # Common database ports
    db_ports = {1433, 1521, 27017, 3306, 5432, 5984, 6379, 9042, 9200}
    
    # Common services that are typically safe
    dns_ports = {53}
    ntp_ports = {123}
    dhcp_ports = {67, 68}
    
    # Combine all safe ports
    safe_ports = web_ports | dns_ports | ntp_ports | dhcp_ports
    
    # Check if this is traffic to/from common safe ports
    if src_port in safe_ports or dst_port in safe_ports:
        # If it's a database attack prediction on a non-database port, it's likely a false positive
        if isinstance(prediction_label, (int, str)):
            prediction_str = str(prediction_label).lower()
            db_keywords = ['postgres', 'mysql', 'sql', 'mssql', 'oracle', 'mongodb', 'redis']
            if any(db in prediction_str for db in db_keywords):
                if dst_port not in db_ports:  # Only if destination isn't a known DB port
                    return True
        return True
    
    # Additional checks based on features if available
    if features:
        # Check for small packet sizes (common in control traffic)
        if 'packet_length' in features and features['packet_length'] < 100:
            return True
            
        # Check for common benign protocols
        if 'protocol' in features:
            # ICMP is often used for network diagnostics
            if features['protocol'] in [1, 58]:  # ICMP, ICMPv6
                return True
    
    return False
            
    # If traffic is on common database ports and matches expected protocols
    if (src_port in db_ports or dst_port in db_ports) and protocol == 'TCP':
        # If the model flagged this as a web attack but it's on a database port,
        # it's likely a false positive
        prediction_str = str(prediction_label).lower()
        if any(web in prediction_str for web in ['http', 'https', 'web', 'xss', 'sql']):
            return True
    
    # If the prediction is for a known benign label
    if str(prediction_label).lower() in ['benign', 'normal', '0']:
        return True
        
    return False

def get_attack_name(label):
    """Convert numeric label to attack name"""
    try:
        if isinstance(label, str) and label.isdigit():
            label = int(label)
        return CIC_ATTACK_TYPES.get(int(label), f'Unknown_Attack_{label}')
    except (ValueError, TypeError):
        return str(label)

# Global variables
DEBUG_HEADER_WRITTEN = False
CSV_HEADER_WRITTEN = False
initialized = False

# Global flow tracker
global_flow_tracker = {}

def get_flow_id(pkt):
    """
    Generate a flow ID based on IP addresses, ports, and protocol
    
    Args:
        pkt: Scapy packet
        
    Returns:
        tuple: (flow_id, is_reverse_flow) where flow_id is a tuple of 
               (src_ip, dst_ip, protocol, src_port, dst_port) and 
               is_reverse_flow indicates if this is a reverse flow
    """
    if pkt.haslayer('IP'):
        src = pkt['IP'].src
        dst = pkt['IP'].dst
        proto = pkt['IP'].proto
    elif pkt.haslayer('IPv6'):
        src = pkt['IPv6'].src
        dst = pkt['IPv6'].dst
        proto = pkt['IPv6'].nh
    else:
        return None, False
    
    # Get ports if available
    src_port = 0
    dst_port = 0
    if pkt.haslayer('TCP'):
        src_port = pkt['TCP'].sport
        dst_port = pkt['TCP'].dport
    elif pkt.haslayer('UDP'):
        src_port = pkt['UDP'].sport
        dst_port = pkt['UDP'].dport
    
    # Create a consistent flow ID (bidirectional)
    if src < dst:
        return (src, dst, proto, src_port, dst_port), False
    else:
        return (dst, src, proto, dst_port, src_port), True

def initialize_flow_tracker():
    """Initialize the global flow tracker"""
    global global_flow_tracker
    global_flow_tracker = {}

def cleanup_old_flows(max_age=300):
    """
    Remove flows that haven't been seen in a while
    
    Args:
        max_age: Maximum age in seconds before a flow is considered stale
        
    Returns:
        int: Number of flows removed
    """
    current_time = time.time()
    expired_flows = [fid for fid, f in global_flow_tracker.items() 
                    if current_time - f.get('last_seen', 0) > max_age]
    
    for fid in expired_flows:
        del global_flow_tracker[fid]
    
    return len(expired_flows)

def get_flow_stats():
    """
    Get statistics about current flows
    
    Returns:
        dict: Dictionary containing flow statistics
    """
    if not global_flow_tracker:
        return {
            'total_flows': 0,
            'total_packets': 0,
            'total_bytes': 0,
            'avg_flow_duration': 0
        }
    
    total_packets = sum(f.get('packet_count', 0) for f in global_flow_tracker.values())
    total_bytes = sum(f.get('byte_count', 0) for f in global_flow_tracker.values())
    
    # Calculate average flow duration
    durations = [f.get('last_seen', 0) - f.get('start_time', 0) 
                for f in global_flow_tracker.values() 
                if 'last_seen' in f and 'start_time' in f]
    
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    return {
        'total_flows': len(global_flow_tracker),
        'total_packets': total_packets,
        'total_bytes': total_bytes,
        'avg_flow_duration': avg_duration
    }
label_counter = Counter()
model = None
label_encoder = None
scaler = None
selected_features = None
processed_packets = 0
last_report_time = time.time()

# Initialize trackers
connection_tracker = None  # Will be initialized in main()
attack_detector = None
packet_processor = None
flow_tracker = defaultdict(lambda: {
    "packet_count": 0,
    "byte_count": 0,
    "timestamps": [],
    "start_time": time.time(),
    "flags": {"SYN": 0, "ACK": 0, "FIN": 0, "RST": 0},
})

# Performance monitoring
performance_stats = {
    'total_packets': 0,
    'processed_packets': 0,
    'dropped_packets': 0,
    'last_report': time.time(),
    'batch_size': 100,  # Process packets in batches
    'attack_detected': 0,
    'last_alert_time': 0
}

# Cache for recent decisions to avoid redundant processing
DECISION_CACHE = {}
CACHE_SIZE = 10000  # Maximum number of cache entries

# Common benign ports for fast filtering
COMMON_PORTS = {
    # Web
    80, 443, 8080, 8443, 8000, 8008, 8081, 8088, 8888,  # HTTP/HTTPS
    # Common services
    21, 22, 23, 25, 53, 67, 68, 69, 110, 123, 137, 138, 139, 143, 161, 162, 389, 443,
    445, 465, 514, 515, 548, 587, 631, 636, 993, 995, 1025, 1026, 1027, 1028, 1029,
    1080, 1194, 1433, 1434, 1521, 1701, 1723, 1900, 2049, 2082, 2083, 2086, 2087,
    2095, 2096, 3000, 3128, 3306, 3389, 4000, 4040, 4369, 4500, 4567, 4711, 4712,
    5000, 5001, 5002, 5003, 5004, 5005, 5060, 5104, 5106, 5222, 5223, 5228, 5353,
    5432, 5601, 5672, 5900, 5938, 5984, 6000, 6379, 6666, 7000, 7077, 7474, 7547,
    7575, 8000, 8005, 8009, 8020, 8042, 8069, 8080, 8081, 8083, 8088, 8089, 8090,
    8091, 8095, 8096, 8100, 8140, 8172, 8181, 8200, 8243, 8280, 8281, 8333,
    8400, 8443, 8500, 8530, 8531, 8880, 8888, 8983, 9000, 9001, 9002, 9042, 9060,
    9080, 9090, 9091, 9092, 9100, 9140, 9160, 9200, 9300, 9418, 9443, 9600, 9800,
    9981, 10000, 10250, 10255, 10443, 11080, 11371, 12018, 12046, 12443, 14000,
    16000, 16992, 16993, 18080, 18081, 18091, 18092, 20000, 27017, 27018, 27019,
    28017, 32400, 32768, 35357, 44818, 47001, 47002, 47808, 49152, 49153, 49154,
    49155, 49156, 49157, 50000, 50010, 50020, 50030, 50060, 50070, 50075, 50090,
    54321, 56000, 56789, 60000, 60010, 60030, 61616, 62078, 64738
}

# Cache for recent decisions to avoid redundant processing
DECISION_CACHE = {}
CACHE_SIZE = 10000  # Maximum number of cache entries

class RateLimiter:
    def __init__(self, max_events=100, time_window=60):
        self.events = []
        self.max_events = max_events
        self.time_window = time_window
    
    def check_rate(self, event_type, identifier):
        """Check if rate limit is exceeded for given event type and identifier"""
        current_time = time.time()
        
        # Remove events older than time_window
        self.events = [ts for ts in self.events if current_time - ts < self.time_window]
        
        # Check if adding a new event would exceed the limit
        if len(self.events) >= self.max_events:
            return False
            
        self.events.append(current_time)
        return True

# Initialize rate limiter for high traffic
rate_limiter = RateLimiter(max_events=5000, time_window=60)  # Increased limit for high traffic


def load_cic_model(model_path: str, encoder_path: str = None, 
                   scaler_path: str = None, features_path: str = None):
    """
    Load CIC-trained model and preprocessing artifacts
    
    Args:
        model_path: Path to trained model (.pkl)
        encoder_path: Path to label encoder (.pkl)
        scaler_path: Path to feature scaler (.pkl)
        features_path: Path to selected features list (.txt)
    """
    global model, label_encoder, scaler, selected_features, model_data
    
    print(f"Loading model from {model_path}...")
    model_data = joblib.load(model_path)
    
    # Extract the actual model and feature names
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        if 'feature_names' in model_data:
            selected_features = model_data['feature_names']
            print(f"Loaded model with {len(selected_features)} features")
    else:
        model = model_data  # In case it's not in a dictionary
    
    # Handle different model formats
    if isinstance(model_data, dict):
        # Try to extract model from common keys
        model = model_data.get('model') or model_data.get('best_model') or model_data
        
        # Try to get label encoder from model data
        if 'label_encoder' in model_data:
            label_encoder = model_data['label_encoder']
        elif 'le' in model_data:
            label_encoder = model_data['le']
            
        # Try to get feature names
        if 'feature_names' in model_data:
            selected_features = model_data['feature_names']
    else:
        model = model_data
    
    # Load additional artifacts if provided
    if encoder_path and os.path.exists(encoder_path) and label_encoder is None:
        print(f"Loading label encoder from {encoder_path}...")
        label_encoder = joblib.load(encoder_path)
    
    if scaler_path:
        if os.path.exists(scaler_path):
            print(f"Loading scaler from {scaler_path}...")
            try:
                scaler = joblib.load(scaler_path)
                print("✓ Scaler loaded successfully")
                # Print scaler details for debugging
                if hasattr(scaler, 'scale_'):
                    print(f"  - Scaler type: {type(scaler).__name__}")
                    print(f"  - Features scaled: {len(scaler.scale_) if hasattr(scaler, 'scale_') else 'N/A'}")
                    print(f"  - Mean: {scaler.mean_[:5] if hasattr(scaler, 'mean_') else 'N/A'}...")
            except Exception as e:
                print(f"⚠️  Failed to load scaler: {e}")
                print("⚠️  Continuing without feature scaling (this may affect model accuracy)")
        else:
            print(f"⚠️  Scaler file not found at {scaler_path}")
            print("⚠️  Continuing without feature scaling (this may affect model accuracy)")
    
    if features_path and os.path.exists(features_path) and not selected_features:
        print(f"Loading selected features from {features_path}...")
        with open(features_path, 'r') as f:
            selected_features = [line.strip() for line in f.readlines()]
    
    print(f"✓ Model loaded successfully (Type: {type(model).__name__})")
    if hasattr(model, 'feature_importances_'):
        print(f"✓ Model has {len(model.feature_importances_)} features")
        
        # Print feature importances if available
        if hasattr(model, 'feature_names_in_'):
            print("\nTop 10 most important features:")
            try:
                importances = list(zip(model.feature_names_in_, model.feature_importances_))
                for feature, importance in sorted(importances, key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {feature}: {importance:.4f}")
            except Exception as e:
                print(f"  Could not display feature importances: {e}")
    
    # Print model feature names if available
    if hasattr(model, 'feature_names_in_'):
        print(f"\nFeature Information:")
        print(f"  - Expected features: {len(model.feature_names_in_)}")
        print(f"  - First 10 features: {', '.join(model.feature_names_in_[:10])}...")
    
    if label_encoder is not None:
        print(f"✓ Label encoder found with {len(label_encoder.classes_)} classes")


def validate_features(df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    """
    Validate that all expected features are present and have valid values.
    
    Args:
        df: DataFrame with current features
        expected_features: List of feature names expected by the model
        
    Returns:
        DataFrame with all expected features, filled with 0 if missing
    """
    missing = set(expected_features) - set(df.columns)
    if missing:
        print(f"⚠️  Adding {len(missing)} missing features with default values")
        for feat in missing:
            df[feat] = 0  # Add missing features with default value 0
    
    # Ensure no NaN values
    if df.isnull().any().any():
        print("⚠️  NaN values detected in features, filling with 0")
        df = df.fillna(0)
    
    # Ensure correct feature order
    return df[expected_features]

def map_pcap_features_to_cic(pcap_features: dict) -> dict:
    """
    Map PCAP-extracted features to CIC-like features with comprehensive handling
    Returns a dictionary with all 54 features expected by the CIC-trained model
    """
    # Initialize with default values for all CIC features
    cic_features = {
        # Basic connection features
        'duration': 0.0,
        'protocol_type': 0,  # Will be mapped to numeric: tcp=0, udp=1, icmp=2
        'service': 0,       # Will be set to destination port
        'flag': 'OTH',      # Connection status flag
        'src_bytes': 0,     # Bytes from source to destination
        'dst_bytes': 0,     # Bytes from destination to source
        'land': 0,          # 1 if connection is from/to the same host/port
        'wrong_fragment': 0,
        'urgent': 0,
        
        # Content features (usually 0 in live traffic)
        'hot': 0,                   # Number of 'hot' indicators
        'num_failed_logins': 0,     # Number of failed login attempts
        'logged_in': 0,             # 1 if successfully logged in; 0 otherwise
        'num_compromised': 0,       # Number of 'compromised' conditions
        'root_shell': 0,            # 1 if root shell is obtained; 0 otherwise
        'su_attempted': 0,          # 1 if 'su root' command attempted; 0 otherwise
        'num_root': 0,              # Number of 'root' accesses
        'num_file_creations': 0,    # Number of file creation operations
        'num_shells': 0,            # Number of shell prompts
        'num_access_files': 0,      # Number of operations on access control files
        'num_outbound_cmds': 0,     # Number of outbound commands in an ftp session
        'is_host_login': 0,         # 1 if the login is from a 'hot' list; 0 otherwise
        'is_guest_login': 0,        # 1 if the login is a 'guest' login; 0 otherwise
        
        # Time-based features
        'count': 1,                 # Number of connections to the same host as the current connection in the past two seconds
        'srv_count': 1,             # Number of connections to the same service as the current connection in the past two seconds
        
        # Host-based features
        'dst_host_count': 1,        # Number of connections having the same destination host
        'dst_host_srv_count': 1,    # Number of connections having the same destination host and service
        
        # Flow features
        'flow_duration': 0.1,       # Duration of the flow in seconds
        'fwd_packets/s': 0.0,       # Forward packets per second
        'bwd_packets/s': 0.0,       # Backward packets per second
        'packet_length': 0,         # Total length of the packet
        'min_packet_length': 0,     # Minimum packet length
        'max_packet_length': 0,     # Maximum packet length
        'fwd_iat_total': 0.0,       # Total time between two packets sent in the forward direction
        'fwd_iat_mean': 0.0,        # Mean time between two packets sent in the forward direction
        'fwd_iat_std': 0.0,         # Standard deviation of time between two packets sent in the forward direction
        'init_win_bytes_forward': 0, # Initial window size in the forward direction
        'min_seg_size_forward': 536, # Minimum segment size in the forward direction
        'flow_bytes/s': 0.0,        # Number of flow bytes per second
        'flow_packets/s': 0.0,      # Number of flow packets per second
        'is_privileged_port': 0,    # 1 if destination port is a privileged port (<1024)
        'is_common_port': 0,        # 1 if destination port is a common service port
        
        # TCP flags (one-hot encoded)
        'tcp_flag_syn': 0,  # SYN flag set
        'tcp_flag_ack': 0,  # ACK flag set
        'tcp_flag_fin': 0,  # FIN flag set
        'tcp_flag_rst': 0,  # RST flag set
        'tcp_flag_psh': 0,  # PSH flag set
        'tcp_flag_urg': 0,  # URG flag set
        
        # Connection flags (one-hot encoded)
        'flag_OTH': 0,  # No status
        'flag_REJ': 0,  # Rejected
        'flag_RSTO': 0, # Reset by originator
        'flag_RSTOS0': 0, # Reset by originator with SYN-ACK
        'flag_RSTR': 0,  # Reset by responder
        'flag_S0': 0,    # Connection attempt seen, no reply
        'flag_S1': 0,    # Connection established, not terminated
        'flag_S2': 0,    # Connection established, originator closed
        'flag_S3': 0,    # Connection established, responder closed
        'flag_SF': 0,    # Normal establishment and termination
        'flag_SH': 0,    # Connection attempt to a port that is not open
        
        # Ports
        'sport': 0,      # Source port
        'dport': 0,      # Destination port
        
        # TTL
        'ttl': 0,        # Time to live
        
        # Additional features for better detection
        'src_ip': '',    # Source IP address
        'dst_ip': '',    # Destination IP address
        'timestamp': 0.0 # Timestamp of the packet
    }
    
    try:
        # Basic packet features
        if 'src_bytes' in pcap_features:
            cic_features['src_bytes'] = max(0, int(pcap_features['src_bytes'] or 0))
        if 'dst_bytes' in pcap_features:
            cic_features['dst_bytes'] = max(0, int(pcap_features['dst_bytes'] or 0))
        if 'packet_length' in pcap_features:
            plen = max(0, int(pcap_features['packet_length'] or 0))
            cic_features['packet_length'] = plen
            cic_features['min_packet_length'] = plen
            cic_features['max_packet_length'] = plen
            
        # Set IP addresses if available
        if 'src_ip' in pcap_features:
            cic_features['src_ip'] = str(pcap_features['src_ip'])
        if 'dst_ip' in pcap_features:
            cic_features['dst_ip'] = str(pcap_features['dst_ip'])
        
        # Protocol handling - map to numeric values expected by the model
        protocol_map = {
            'tcp': 0, 'tcp6': 0, '6': 0,  # TCP
            'udp': 1, 'udp6': 1, '17': 1,  # UDP
            'icmp': 2, 'icmp6': 2, '1': 2  # ICMP
        }
        
        # Determine protocol
        protocol = None
        if 'protocol' in pcap_features and pcap_features['protocol']:
            protocol = str(pcap_features['protocol']).lower()
        elif pcap_features.get('protocol_type_tcp') == 1 or pcap_features.get('tcp'):
            protocol = 'tcp'
        elif pcap_features.get('protocol_type_udp') == 1 or pcap_features.get('udp'):
            protocol = 'udp'
        elif pcap_features.get('protocol_type_icmp') == 1 or pcap_features.get('icmp'):
            protocol = 'icmp'
            
        # Set protocol type (0=TCP, 1=UDP, 2=ICMP)
        cic_features['protocol_type'] = protocol_map.get(protocol, 0)
        
        # Port handling with validation
        def get_port(port_key, default=0):
            port = pcap_features.get(port_key, default)
            try:
                port = int(port) if port is not None else default
                # Ensure port is within valid range
                return max(0, min(65535, port))
            except (ValueError, TypeError):
                return default
        
        # Get source and destination ports from various possible keys
        sport = get_port('sport', 
                       get_port('tcp_sport', 
                              get_port('udp_sport', 
                                     get_port('src_port', 0))))
                                      
        dport = get_port('dport', 
                       get_port('tcp_dport', 
                              get_port('udp_dport', 
                                     get_port('dst_port', 0))))
        
        # Set port-related features
        cic_features['sport'] = sport
        cic_features['dport'] = dport
        cic_features['service'] = dport  # Service is typically the destination port
        
        # Check for land attack (source and destination IP and port are the same)
        if ('src_ip' in pcap_features and 'dst_ip' in pcap_features and 
            pcap_features['src_ip'] == pcap_features['dst_ip'] and 
            sport == dport):
            cic_features['land'] = 1
        
        # TCP flags with enhanced detection
        tcp_flags = {
            'fin': int(bool(pcap_features.get('tcp_flag_FIN') or pcap_features.get('tcp.fin', 0))),
            'syn': int(bool(pcap_features.get('tcp_flag_SYN') or pcap_features.get('tcp.syn', 0))),
            'rst': int(bool(pcap_features.get('tcp_flag_RST') or pcap_features.get('tcp.rst', 0))),
            'psh': int(bool(pcap_features.get('tcp_flag_PSH') or pcap_features.get('tcp.psh', 0))),
            'ack': int(bool(pcap_features.get('tcp_flag_ACK') or pcap_features.get('tcp.ack', 0))),
            'urg': int(bool(pcap_features.get('tcp_flag_URG') or pcap_features.get('tcp.urg', 0)))
        }
        
        # Update TCP flags in features
        for flag, value in tcp_flags.items():
            cic_features[f'tcp_flag_{flag}'] = value
        
        # Set connection flag based on TCP state
        if tcp_flags['syn'] and tcp_flags['ack']:
            flag = 'SF'      # Established connection (SYN-ACK)
            cic_features['logged_in'] = 1  # Consider this a successful login for the model
        elif tcp_flags['syn']:
            flag = 'S0'      # Connection attempt (SYN)
        elif tcp_flags['fin'] and tcp_flags['ack']:
            flag = 'SF'      # Graceful connection termination
        elif tcp_flags['rst']:
            flag = 'RSTO'    # Connection reset
        elif tcp_flags['fin']:
            flag = 'SF'      # Connection termination
        else:
            flag = 'OTH'     # Other cases (no flags set or just ACK/PSH/URG)
        
        cic_features['flag'] = flag
        
        # Set flag features (one-hot encoded)
        flag_types = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
        for f in flag_types:
            cic_features[f'flag_{f}'] = 1 if flag == f else 0
        
        # Flow features with enhanced calculations
        flow_duration = max(0.001, float(pcap_features.get('flow_duration', 0.1)))
        packet_count = max(1, int(pcap_features.get('packet_count', 1)))
        
        # Set timing features
        cic_features['flow_duration'] = flow_duration
        cic_features['duration'] = flow_duration
        
        # Calculate rates and intervals
        if flow_duration > 0:
            # Packet rates
            packet_rate = packet_count / flow_duration
            cic_features['fwd_packets/s'] = packet_rate
            cic_features['flow_packets/s'] = packet_rate
            
            # Inter-arrival times
            if packet_count > 1:
                iat_mean = flow_duration / (packet_count - 1) if packet_count > 1 else 0
                cic_features['fwd_iat_mean'] = iat_mean
                cic_features['fwd_iat_total'] = flow_duration
                
                # Estimate standard deviation (simplified)
                cic_features['fwd_iat_std'] = iat_mean * 0.5  # Approximation
            
            # Byte rates
            total_bytes = cic_features['src_bytes'] + cic_features['dst_bytes']
            if total_bytes > 0:
                byte_rate = total_bytes / flow_duration
                cic_features['flow_bytes/s'] = byte_rate
                
                # Set some content-based features based on byte counts
                if total_bytes > 10000:  # Large transfer
                    cic_features['hot'] = 1
                
                # If we have more bytes in response than request, might indicate file download
                if cic_features['dst_bytes'] > cic_features['src_bytes'] * 10:
                    cic_features['num_file_creations'] = 1
        
        # Window size and TCP options
        if 'tcp_window' in pcap_features:
            win_size = int(pcap_features['tcp_window'] or 0)
            cic_features['init_win_bytes_forward'] = win_size
            
            # Set segment size based on window size (simplified)
            if win_size > 0:
                mss = min(win_size, 1460)  # Typical MSS is up to 1460 bytes
                cic_features['min_seg_size_forward'] = mss
        
        # Port-based features
        if 0 < dport < 1024:
            cic_features['is_privileged_port'] = 1
            
            # Common services that might indicate specific behaviors
            if dport in {21, 22, 23, 25, 110, 143, 445, 993, 995}:
                cic_features['num_failed_logins'] = 1  # Potential login service
            elif dport in {20, 21, 22, 69, 115, 161, 162, 389, 443, 445, 636, 989, 990, 992, 993, 995, 1433, 1521, 2049, 3306, 3389, 5432, 5800, 5900, 6000, 8000, 8080, 8443, 8888}:
                cic_features['is_common_port'] = 1
        
        # TTL analysis
        if 'ttl' in pcap_features and pcap_features['ttl'] is not None:
            try:
                ttl = int(pcap_features['ttl'])
                cic_features['ttl'] = ttl
                
                # TTL-based OS fingerprinting (simplified)
                if ttl <= 64:
                    cic_features['is_host_login'] = 1  # Likely Unix/Linux
                elif ttl <= 128:
                    cic_features['is_guest_login'] = 1  # Likely Windows
                
            except (ValueError, TypeError):
                pass
                
        # Timestamp for tracking
        cic_features['timestamp'] = time.time()
        
<<<<<<< HEAD
        # Debug: Print non-zero features with better formatting
        debug_interval = 5  # seconds between debug outputs
        current_time = time.time()
        
        if 'last_debug_output' not in map_pcap_features_to_cic.__dict__:
            map_pcap_features_to_cic.last_debug_output = 0
            
        if current_time - map_pcap_features_to_cic.last_debug_output > debug_interval:
            map_pcap_features_to_cic.last_debug_output = current_time
            
            # Group features by category for better readability
            feature_categories = {
                'Basic': ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land'],
                'Content': ['hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 
                           'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files'],
                'Time': ['count', 'srv_count', 'flow_duration', 'fwd_iat_mean', 'fwd_iat_total'],
                'Host': ['dst_host_count', 'dst_host_srv_count'],
                'Flow': ['fwd_packets/s', 'flow_packets/s', 'flow_bytes/s', 'packet_length'],
                'TCP': ['tcp_flag_syn', 'tcp_flag_ack', 'tcp_flag_fin', 'tcp_flag_rst', 'tcp_flag_psh', 'tcp_flag_urg'],
                'Ports': ['sport', 'dport', 'is_privileged_port', 'is_common_port'],
                'Other': ['ttl', 'src_ip', 'dst_ip']
            }
            
            print("\n" + "="*80)
            print("FEATURE MAPPING DEBUG (non-zero values)")
            print("="*80)
            
            for category, features in feature_categories.items():
                cat_features = {k: v for k, v in cic_features.items() 
                              if k in features and v != 0 and not (isinstance(v, (int, float)) and v == 0)}
                if cat_features:
                    print(f"\n[{category.upper()} FEATURES]")
                    for k, v in sorted(cat_features.items()):
                        if isinstance(v, float):
                            print(f"  {k:<25}: {v:.4f}")
                        else:
                            print(f"  {k:<25}: {v}")
            
            print("\n" + "="*80 + "\n")
        
=======
        # Debug: Print non-zero features (reduced frequency)
        if int(time.time()) % 30 == 0:  # Print every 30 seconds
            non_zero = {k: v for k, v in cic_features.items() if v != 0 and not k.startswith('flag_') and k != 'flag'}
            print(f"\n--- Non-zero CIC features ({len(non_zero)}/{len(cic_features)}) ---")
            for k, v in sorted(non_zero.items()):
                print(f"  {k}: {v}")

>>>>>>> org/master
    except Exception as e:
        print(f"Error in map_pcap_features_to_cic: {e}")
        import traceback
        traceback.print_exc()

    return cic_features


def predict_with_cic_model(pcap_features: dict) -> tuple:
    """
    Predict using CIC-trained model with improved feature validation and whitelist checks
    
    Args:
        pcap_features: Features extracted from PCAP
    
    Returns:
        Tuple of (predicted_label, confidence)
    """
    global model, label_encoder, scaler, selected_features, model_data
    
    # Get the actual model and feature names
    actual_model = model_data['model'] if isinstance(model_data, dict) and 'model' in model_data else model
    feature_names = model_data.get('feature_names', [f'feature_{i}' for i in range(54)]) if isinstance(model_data, dict) else selected_features
    
    # Extract common ports and protocol for whitelist check
    src_port = int(pcap_features.get('sport', pcap_features.get('src_port', 0)))
    dst_port = int(pcap_features.get('dport', pcap_features.get('dst_port', 0)))
    protocol = pcap_features.get('protocol', '').lower()
    
    # Check if this is likely benign traffic before running model prediction
    if is_benign_port(src_port) or is_benign_port(dst_port):
        # For common ports, we can be more aggressive with whitelisting
        if protocol in ['tcp', 'udp'] and (src_port in [80, 443, 22, 53] or dst_port in [80, 443, 22, 53]):
            if int(time.time()) % 10 == 0:  # Don't spam
                print(f"ℹ️  Traffic on common port {dst_port} (proto: {protocol}) whitelisted")
            return "benign_common_port", 0.0
    
    try:
        # Debug: Print raw features
        if int(time.time()) % 10 == 0:  # Print every 10 seconds to avoid spam
            print("\n--- Raw Packet Features ---")
            for k, v in sorted(pcap_features.items()):
                if v and v != 0:  # Only show non-zero/non-empty values
                    print(f"  {k}: {v}")
        
        # Map PCAP features to CIC format
        cic_features = map_pcap_features_to_cic(pcap_features)
        
        # Create a new dictionary with features in the expected order
        ordered_features = {}
        for i, feat_name in enumerate(feature_names):
            # Try to get the feature value, use 0.0 as default
            # For HTTPS traffic, ensure we don't have false PostgreSQL indicators
            if dst_port == 443 or src_port == 443:
                # Reset features that might cause false PostgreSQL detection
                if feat_name in ['feature_27', 'feature_28', 'feature_45']:
                    ordered_features[feat_name] = 0.0
                else:
                    ordered_features[feat_name] = float(cic_features.get(f'feature_{i}', 0.0))
            else:
                ordered_features[feat_name] = float(cic_features.get(f'feature_{i}', 0.0))
        
        # Create DataFrame with features in the correct order
        df = pd.DataFrame([ordered_features])
        
        # Ensure no NaN values
        if df.isnull().any().any():
            print("⚠️  NaN values detected in features, filling with 0")
            df = df.fillna(0)
            
        # Debug: Print first few features for verification
        if int(time.time()) % 10 == 0:  # Print every 10 seconds
            print("\n--- First 5 feature values ---")
            for i, (feat, val) in enumerate(zip(feature_names[:5], df.iloc[0][:5])):
                print(f"  {feat}: {val}")
        
        # Debug: Print feature summary
        if int(time.time()) % 10 == 0:  # Print every 10 seconds
            print(f"\n--- Feature Summary ---")
            print(f"Total features: {len(df.columns)}")
            print(f"First 10 features: {df.columns.tolist()[:10]}...")
            
            # Print non-zero feature values
            non_zero = {k: v for k, v in df.iloc[0].items() if v != 0}
            print(f"Non-zero features ({len(non_zero)}):")
            for k, v in sorted(non_zero.items()):
                print(f"  {k}: {v}")
        
        # Scale features if scaler is available
        if scaler is not None:
            try:
                # Ensure the input has the same features as the scaler was trained on
                if hasattr(scaler, 'feature_names_in_'):
                    # Reorder columns to match scaler's expected input
                    missing_cols = set(scaler.feature_names_in_) - set(df.columns)
                    if missing_cols:
                        print(f"⚠️  Adding {len(missing_cols)} missing features with default values")
                        for col in missing_cols:
                            df[col] = 0
                    df = df[scaler.feature_names_in_]
                
                # Apply scaling
                scaled_data = scaler.transform(df)
                df = pd.DataFrame(scaled_data, columns=df.columns)
                
                if int(time.time()) % 10 == 0:  # Print every 10 seconds to avoid spam
                    print("\n--- Scaled Feature Summary ---")
                    print(f"Scaled features: {df.shape[1]}")
                    non_zero = {k: v for k, v in df.iloc[0].items() if abs(v) > 1e-6}
                    print(f"Non-zero scaled features ({len(non_zero)}):")
                    for k, v in sorted(non_zero.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                        print(f"  {k}: {v:.4f}")
                    
            except Exception as e:
                print(f"Error scaling features: {e}")
                import traceback
                print(traceback.format_exc())
                return "scaling_error", 0.0
                
        # Final feature validation before prediction
        try:
            # Get expected features from the model
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
            elif selected_features:
                expected_features = selected_features
            else:
                expected_features = df.columns.tolist()
                
            # Validate features before prediction
            df = validate_features(df, expected_features)
            
            # Double-check for NaN values after validation
            if df.isnull().any().any():
                print("⚠️  Warning: NaN values found after validation, filling with 0")
                df = df.fillna(0)
                
        except Exception as e:
            print(f"Error during final feature validation: {e}")
            import traceback
            print(traceback.format_exc())
            return "feature_validation_error", 0.0
        
        # Make prediction
        try:
            # Feature validation is now done before this point
            
            # Get source and destination ports for whitelist check
            src_port = int(pcap_features.get('sport', pcap_features.get('src_port', 0)))
            dst_port = int(pcap_features.get('dport', pcap_features.get('dst_port', 0)))
            protocol = pcap_features.get('protocol', '').lower()
            
            # Try predict_proba first (for classifiers that support it)
            if hasattr(actual_model, 'predict_proba'):
                try:
                    # Convert to numpy array to avoid feature name issues
                    X = df.values
                    proba = actual_model.predict_proba(X)[0]
                except Exception as e:
                    print(f"Error in predict_proba: {e}")
                    # Fall back to predict if predict_proba fails
                    pred = actual_model.predict(X)[0]
                    return pred, 1.0
                pred_idx = proba.argmax()
                confidence = float(proba[pred_idx])
                
                # Get class label
                if hasattr(actual_model, 'classes_'):
                    pred = actual_model.classes_[pred_idx]
                else:
                    pred = pred_idx
                
                # Only apply whitelist for very specific cases
                if is_benign_traffic(pred, src_port, dst_port, protocol, pcap_features):
                    # Only whitelist if we're not very confident AND it's a common service
                    common_services = [80, 443, 53, 22]
                    if (confidence < 0.8 and 
                        (dst_port in common_services or src_port in common_services)):
                        if int(time.time()) % 10 == 0:  # Don't spam
                            print(f"ℹ️  Low confidence benign traffic on port {dst_port} (confidence: {confidence*100:.1f}%)")
                        return "benign_whitelisted", 0.0
                
                # Debug: Print prediction details
                if int(time.time()) % 5 == 0:  # Print every 5 seconds
                    print("\n--- Prediction ---")
                    print(f"Class: {pred} ({get_attack_name(pred)})")
                    print(f"Confidence: {confidence*100:.1f}%")
                    
                    # Show top predictions
                    top_n = min(5, len(proba))
                    top_indices = (-proba).argsort()[:top_n]
                    print("Top predictions:")
                    for i in top_indices:
                        print(f"  {i}. {get_attack_name(i)}: {proba[i]*100:.1f}%")
                
                # Adjust confidence threshold based on prediction
                min_confidence = 0.7  # Default confidence threshold (70%)
                
                # Increase threshold for specific attack types that often have false positives
                if pred == 28:  # PostgreSQL
                    min_confidence = 0.95  # Require 95% confidence for PostgreSQL
                
                # If prediction is below threshold, mark as benign
                if confidence < min_confidence:
                    if int(time.time()) % 10 == 0:  # Don't spam
                        print(f"ℹ️  Low confidence prediction ({confidence*100:.1f}% < {min_confidence*100}%), marking as benign")
                    return "benign_low_confidence", 0.0
                    
                # Additional check for specific attack types that often have false positives
                prediction_str = str(pred).lower()
                if 'postgres' in prediction_str and confidence < 0.95:  # Require higher confidence for DB attacks
                    if int(time.time()) % 10 == 0:
                        print(f"⚠️  Database attack prediction requires higher confidence ({confidence*100:.1f}% < 95%)")
                    return "benign_low_confidence_db", 0.0
                
                return pred, confidence
                
            else:
                # Fall back to predict if predict_proba not available
                pred = actual_model.predict(df[expected_features])[0]
                if int(time.time()) % 5 == 0:  # Print every 5 seconds
                    print(f"\n--- Prediction (no probabilities) ---")
                    print(f"Class: {pred} ({get_attack_name(pred)})")
                return pred, 1.0  # Assume 100% confidence if no probabilities
                
        except Exception as e:
            print(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return "prediction_error", 0.0
            
    except Exception as e:
        print(f"Error in predict_with_cic_model: {e}")
        import traceback
        traceback.print_exc()
        return "error", 0.0


def predict_with_pcap_model(pcap_features: dict, pcap_model, pcap_encoder) -> str:
    """
    Predict using original PCAP-trained model
    
    Args:
        pcap_features: Features extracted from PCAP
        pcap_model: PCAP-trained model
        pcap_encoder: PCAP label encoder
    
    Returns:
        Predicted label
    """
    try:
        df = pd.DataFrame([pcap_features])
        
        if 'label' in df.columns:
            df = df.drop(columns=['label'])
        
        # Expected columns for PCAP model
        expected_columns = [
            "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", 
            "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", 
            "su_attempted", "num_root", "num_file_creations", "num_shells",
            "num_access_files", "is_host_login", "is_guest_login", "count", "srv_count", 
            "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", 
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
            "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate", "protocol_type_icmp", "protocol_type_tcp", 
            "protocol_type_udp", "service_domain_u", "service_eco_i", "service_ecr_i", 
            "service_finger", "service_ftp", "service_ftp_data", "service_http",
            "service_other", "service_private", "service_rare", "service_smtp", 
            "service_telnet", "flag_OTH", "flag_REJ", "flag_RSTO", "flag_RSTOS0", 
            "flag_RSTR", "flag_S0", "flag_S1", "flag_S2", "flag_S3", "flag_SF", "flag_SH"
        ]
        
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[expected_columns]
        
        prediction = pcap_model.predict(df)[0]
        
        # Label mapping for PCAP model
        label_mapping = {
            0: "back", 1: "buffer_overflow", 2: "ftp_write", 3: "guess_passwd",
            4: "imap", 5: "ipsweep", 6: "land", 7: "loadmodule", 8: "multihop",
            9: "neptune", 10: "nmap", 11: "normal", 12: "perl", 13: "phf",
            14: "pod", 15: "portsweep", 16: "rootkit", 17: "satan", 18: "smurf",
            19: "spy", 20: "teardrop", 21: "warezclient", 22: "warezmaster"
        }
        
        label = label_mapping.get(prediction, str(prediction))
        return label
        
    except Exception as e:
        logging.error(f"PCAP model prediction error: {e}")
        return "unknown"


def log_debug_info(features, prediction_label):
    """Log features and predictions to CSV"""
    global DEBUG_HEADER_WRITTEN
    
    features_with_prediction = dict(features)
    features_with_prediction["prediction"] = prediction_label
    
    with open(DEBUG_LOG_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=features_with_prediction.keys())
        if not DEBUG_HEADER_WRITTEN:
            writer.writeheader()
            DEBUG_HEADER_WRITTEN = True
        writer.writerow(features_with_prediction)


# Enhanced scan detection configuration
scan_detection = {
    'port_scan_threshold': 5,  # Reduced threshold for faster detection
    'time_window': 60,         # seconds
    'port_scan_types': {
        'horizontal': True,    # Multiple ports on single host
        'vertical': True,      # Single port on multiple hosts
        'distributed': True    # Multiple sources to multiple ports
    },
    'recent_scans': defaultdict(lambda: {'ports': set(), 'count': 0, 'first_seen': time.time(), 'targets': set()}),
    'last_cleanup': time.time(),
    'port_scan_window': 5,     # Time window in seconds
    'last_scan_alert': {}      # Track last alert time per source IP
}

class FastConnectionTracker:
    """Optimized connection tracker for high-performance monitoring"""
    def __init__(self):
        self.connections = {}  # key: (src_ip, dst_ip, protocol, src_port, dst_port)
        self.timeout = 300     # 5 minutes
        self.last_cleanup = time.time()
        self.cleanup_interval = 60  # Cleanup every 60 seconds
        
    def update(self, features):
        """Update connection tracking with packet features"""
        key = self._connection_key(features)
        current_time = time.time()
        
        # Periodic cleanup
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup(current_time)
            self.last_cleanup = current_time
        
        # Update or create connection
        if key not in self.connections:
            self.connections[key] = {
                'start_time': current_time,
                'last_seen': current_time,
                'packet_count': 1,
                'bytes_sent': features.get('length', 0),
                'flags': set()
            }
        else:
            conn = self.connections[key]
            conn['last_seen'] = current_time
            conn['packet_count'] += 1
            conn['bytes_sent'] += features.get('length', 0)
            
        # Track TCP flags if available
        if 'tcp_flags' in features:
            flags = features['tcp_flags']
            for flag, is_set in flags.items():
                if is_set:
                    self.connections[key]['flags'].add(flag)
    
    def _connection_key(self, features):
        """Create a unique key for a connection"""
        return (
            features.get('src_ip', ''),
            features.get('dst_ip', ''),
            features.get('protocol', 0),
            features.get('sport', 0),
            features.get('dport', 0)
        )
        
    def _cleanup(self, current_time):
        """Remove stale connections"""
        timeout = self.timeout
        stale = [k for k, v in self.connections.items() 
                if current_time - v['last_seen'] > timeout]
        
        # Use dict comprehension for faster cleanup
        if stale:
            self.connections = {k: v for k, v in self.connections.items() 
                              if k not in stale}

class ConnectionTracker(FastConnectionTracker):
    """Legacy compatibility class - uses FastConnectionTracker implementation"""
    pass

class RateLimiter:
    def __init__(self, max_events=100, time_window=60):
        self.events = []
        self.max_events = max_events
        self.time_window = time_window
        
    def check_rate(self, event_type, identifier):
        """Check if rate limit is exceeded for given event type and identifier"""
        current_time = time.time()
        self.events = [t for t in self.events 
                      if current_time - t[2] < self.time_window]
        
        count = sum(1 for e in self.events 
                   if e[0] == event_type and e[1] == identifier)
                   
        if count >= self.max_events:
            return False
            
        self.events.append((event_type, identifier, current_time))
        return True

def analyze_http(payload):
    """Analyze HTTP traffic for suspicious patterns"""
    if not payload or not isinstance(payload, str):
        return None
        
    payload = payload.lower()
    suspicious_patterns = [
        ('SQL_INJECTION', ["' or '1'='1", 'union select', 'drop table', 'select * from']),
        ('XSS', ['<script>', 'javascript:', 'onerror=', 'alert(']),
        ('LFI', ['../../', '/etc/passwd', 'win.ini', 'boot.ini']),
        ('COMMAND_INJECTION', [';ls', '|cat ', '`id`', '$(whoami)']),
        ('DIRECTORY_TRAVERSAL', ['../', '..%2f', '..\\', '%2e%2e/'])
    ]
    
    for vuln_type, patterns in suspicious_patterns:
        if any(p in payload for p in patterns):
            return f"HTTP_{vuln_type}"
    return None

def analyze_dns(query):
    """Analyze DNS traffic for tunneling or exfiltration"""
    if not query or not isinstance(query, str):
        return None
        
    query = query.lower()
    
    # Check for suspicious DNS patterns
    if len(query) > 50:  # Unusually long domain
        return "DNS_TUNNELING"
    if any(x in query for x in ['tunnel', 'exfil', 'malware', 'command', 'c2', 'cnc']):
        return "DNS_EXFILTRATION"
    if sum(c.isdigit() for c in query) > 5:  # Excessive numbers in domain
        return "DNS_EXFILTRATION"
    return None

def is_port_scan_enhanced(packet):
    """Enhanced port scan detection with multiple scan types"""
    src_ip = packet.get('src_ip')
    dst_ip = packet.get('dst_ip')
    dst_port = packet.get('dport')
    protocol = packet.get('protocol', '').lower()
    
    if not all([src_ip, dst_ip, dst_port, protocol in ['tcp', 'udp']]):
        return False
        
    current_time = time.time()
    
    # Clean up old entries periodically
    if current_time - scan_detection['last_cleanup'] > 60:  # Every minute
        for key in list(scan_detection['recent_scans'].keys()):
            if current_time - scan_detection['recent_scans'][key]['first_seen'] > scan_detection['time_window']:
                del scan_detection['recent_scans'][key]
        scan_detection['last_cleanup'] = current_time
    
    # Track horizontal scans (multiple ports on single host)
    if scan_detection['port_scan_types']['horizontal']:
        h_key = f"{src_ip}-{dst_ip}"
        if h_key not in scan_detection['recent_scans']:
            scan_detection['recent_scans'][h_key] = {
                'ports': set(),
                'count': 0,
                'first_seen': current_time,
                'targets': set()
            }
        
        scan_info = scan_detection['recent_scans'][h_key]
        if dst_port not in scan_info['ports']:
            scan_info['ports'].add(dst_port)
            scan_info['count'] += 1
            
            if len(scan_info['ports']) >= scan_detection['port_scan_threshold']:
                return f"HORIZONTAL_PORT_SCAN from {src_ip} to {dst_ip} ({len(scan_info['ports'])} ports)"
    
    # Track vertical scans (single port on multiple hosts)
    if scan_detection['port_scan_types']['vertical'] and dst_port > 0:
        v_key = f"{src_ip}-{dst_port}"
        if v_key not in scan_detection['recent_scans']:
            scan_detection['recent_scans'][v_key] = {
                'targets': set(),
                'count': 0,
                'first_seen': current_time,
                'ports': set()
            }
        
        scan_info = scan_detection['recent_scans'][v_key]
        if dst_ip not in scan_info['targets']:
            scan_info['targets'].add(dst_ip)
            scan_info['count'] += 1
            
            if len(scan_info['targets']) >= scan_detection['port_scan_threshold']:
                return f"VERTICAL_PORT_SCAN from {src_ip} (port {dst_port} to {len(scan_info['targets'])} hosts)"

    return False

def is_benign_port(port):
    """Quick check if port is in the common benign ports list"""
    return port in COMMON_PORTS

def update_performance_stats(processed=True):
    """Update performance statistics and print periodic report"""
    global performance_stats, last_report_time, processed_packets, attack_detector
    
    current_time = time.time()
    time_since_last_report = current_time - performance_stats['last_report']
    
    # Print report every 2 seconds for better responsiveness
    if time_since_last_report >= 2:
        total = performance_stats['total_packets']
        processed = performance_stats['processed_packets']
        dropped = performance_stats['dropped_packets']
        attacks = performance_stats['attack_detected']
        
        # Calculate rates
        pps = total / time_since_last_report if time_since_last_report > 0 else 0
        drop_rate = (dropped / total * 100) if total > 0 else 0
        attack_rate = attacks / time_since_last_report if time_since_last_report > 0 else 0
        
        # Get queue size if packet processor is active
        queue_size = packet_processor.packet_queue.qsize() if packet_processor else 0
        
        # Print status line
        status = (
            f"\r[+] PPS: {pps:6.1f} | "
            f"Proc: {processed:6d} | "
            f"Drop: {dropped:4d} ({drop_rate:4.1f}%) | "
            f"Attacks/s: {attack_rate:4.1f} | "
            f"Queue: {queue_size:4d} | "
            f"Cache: {len(DECISION_CACHE):5d}/{CACHE_SIZE}"
        )
        
        # Add color based on load
        if pps > 1000:
            status = f"\033[93m{status}\033[0m"  # Yellow for high load
        elif pps > 5000:
            status = f"\033[91m{status}\033[0m"  # Red for very high load
            
        print(status, end='', flush=True)
        
        # Reset counters
        performance_stats.update({
            'total_packets': 0,
            'processed_packets': 0,
            'dropped_packets': 0,
            'attack_detected': 0,
            'last_report': current_time
        })

def initialize_detection_components(model_path=None, scaler_path=None, feature_columns=None):
    """Initialize attack detection components"""
    global attack_detector, packet_processor

    if not ENHANCED_DETECTION_AVAILABLE:
        print("[!] Enhanced detection not available, skipping initialization")
        return

    # Configure attack detector
    config = {
        'model_path': model_path,
        'scaler_path': scaler_path,
        'feature_columns': feature_columns or []
    }

    attack_detector = AdvancedAttackDetector(config)
    packet_processor = OptimizedPacketProcessor(
        detector=attack_detector,
        batch_size=100,
        max_workers=4
    )
    packet_processor.start()

    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down packet processor...")
        if packet_processor:
            packet_processor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


def packet_handler(pkt, use_cic_model=True, pcap_model=None, pcap_encoder=None):
    """
    Handle each captured packet with performance optimizations and enhanced attack detection
    
    Args:
        pkt: Scapy packet
        use_cic_model: Whether to use CIC-trained model
        pcap_model: PCAP-trained model (if not using CIC)
        pcap_encoder: PCAP label encoder
        
    Returns:
        bool: True to continue sniffing, False to stop
    """
    global global_flow_tracker, performance_stats, initialized, CSV_HEADER_WRITTEN, label_counter, connection_tracker, rate_limiter, DECISION_CACHE
    
    # Initialize performance stats if needed
    if 'total_packets' not in performance_stats:
        performance_stats = {
            'total_packets': 0,
            'processed_packets': 0,
            'dropped_packets': 0,
            'last_report': time.time(),
            'batch_size': 50,
            'batch_count': 0,
            'last_batch_time': time.time(),
            'attack_detected': 0,
            'errors': 0,
            'last_alert_time': 0,
            'last_packet_time': time.time(),
            'predictions': {},
            'last_prediction_time': 0,
            'flow_stats': {
                'total_flows': 0,
                'total_packets': 0,
                'total_bytes': 0,
                'avg_flow_duration': 0,
                'flows_cleaned': 0
            }
        }
    
    # Update total packets counter
    performance_stats['total_packets'] += 1
    current_time = time.time()
    performance_stats['last_packet_time'] = current_time
    
    # Print a dot for each packet to show activity (but limit to 100 dots per line)
    if performance_stats['total_packets'] % 10 == 0:
        print('\r' + '.' * (performance_stats['total_packets'] // 10 % 100), end='', flush=True)
    
    # Periodic cleanup of old flows (every 60 seconds)
    if current_time - performance_stats.get('last_flow_cleanup', 0) > 60:
        cleaned = cleanup_old_flows()
        if cleaned > 0:
            performance_stats['flow_stats']['flows_cleaned'] += cleaned
        performance_stats['last_flow_cleanup'] = current_time
        
        # Update flow stats
        flow_stats = get_flow_stats()
        performance_stats['flow_stats'].update(flow_stats)
    
    # Get flow ID for this packet
    flow_id, is_reverse = get_flow_id(pkt)
    if flow_id is None:
        performance_stats['dropped_packets'] += 1
        return True  # Skip non-IP packets
    
    # Initialize flow if it doesn't exist
    if flow_id not in global_flow_tracker:
        global_flow_tracker[flow_id] = {
            'start_time': current_time,
            'last_seen': current_time,
            'packet_count': 0,
            'byte_count': 0,
            'packet_lengths': [],
            'interarrival_times': [],
            'last_packet_time': current_time,
            'flags': [],
            'src_ports': set(),
            'dst_ports': set(),
            'services': set(),
            'is_reverse': is_reverse
        }
        performance_stats['flow_stats']['total_flows'] += 1
    
    # Update flow statistics
    flow = global_flow_tracker[flow_id]
    flow['last_seen'] = current_time
    flow['packet_count'] += 1
    flow['byte_count'] += len(pkt)
    flow['packet_lengths'].append(len(pkt))
    
    # Calculate inter-arrival time
    if 'last_packet_time' in flow:
        iat = current_time - flow['last_packet_time']
        flow['interarrival_times'].append(iat)
    flow['last_packet_time'] = current_time
    
    # Extract TCP flags if available
    if pkt.haslayer('TCP'):
        tcp = pkt['TCP']
        flow['flags'].append(tcp.flags)
        
        # Track source and destination ports
        flow['src_ports'].add(tcp.sport)
        flow['dst_ports'].add(tcp.dport)
        
        # Track services based on destination port
        if tcp.dport < 1024:  # Well-known ports
            service = {
                80: 'http',
                443: 'https',
                22: 'ssh',
                21: 'ftp',
                25: 'smtp',
                53: 'dns',
                3306: 'mysql',
                5432: 'postgresql',
                27017: 'mongodb'
            }.get(tcp.dport, f'port_{tcp.dport}')
            flow['services'].add(service)
    
    # Update performance stats
    performance_stats['flow_stats']['total_packets'] += 1
    performance_stats['flow_stats']['total_bytes'] += len(pkt)
    
    # Print periodic summary (every 100 packets)
    if performance_stats['total_packets'] % 100 == 0:
        print(f"\n[+] Flow Stats: {performance_stats['flow_stats']['total_flows']} flows | "
              f"{performance_stats['flow_stats']['total_packets']:,} packets | "
              f"{performance_stats['flow_stats']['total_bytes']:,} bytes | "
              f"{performance_stats['flow_stats']['flows_cleaned']} flows cleaned")
    
    # Skip further processing for now - we'll add prediction in the next step
    return True
    global performance_stats, initialized, CSV_HEADER_WRITTEN, label_counter, connection_tracker, rate_limiter, DECISION_CACHE
    
    # Initialize performance stats if needed
    if 'total_packets' not in performance_stats:
        performance_stats = {
            'total_packets': 0,
            'processed_packets': 0,
            'dropped_packets': 0,
            'last_report': time.time(),
            'batch_size': 50,
            'batch_count': 0,
            'last_batch_time': time.time(),
            'attack_detected': 0,
            'errors': 0,
            'last_alert_time': 0,
            'last_packet_time': time.time(),
            'predictions': {},
            'last_prediction_time': 0
        }
    
    # Update total packets counter
    performance_stats['total_packets'] += 1
<<<<<<< HEAD
    current_time = time.time()
    performance_stats['last_packet_time'] = current_time
    
    # Print a dot for each packet to show activity (but limit to 100 dots per line)
    if performance_stats['total_packets'] % 10 == 0:
        print('\r' + '.' * (performance_stats['total_packets'] // 10 % 100), end='', flush=True)
    
    # Debug: Print packet info for first few packets and periodically
    debug_interval = 2.0  # seconds between debug outputs
    if (performance_stats['total_packets'] <= 5 or 
        (current_time - performance_stats.get('last_debug_output', 0) > debug_interval)):
        
        performance_stats['last_debug_output'] = current_time
        
        # Clear the line and print status
        print('\r' + ' ' * 100 + '\r', end='')
        
        # Basic stats
        elapsed = current_time - performance_stats.get('start_time', current_time)
        pps = performance_stats['total_packets'] / elapsed if elapsed > 0 else 0
        
        print(f"[+] Packets: {performance_stats['total_packets']:,} | "
              f"Rate: {pps:,.1f} pps | "
              f"Attacks: {performance_stats.get('attack_detected', 0)}", end='')
        
        # Print last attack if any
        if 'last_attack' in performance_stats and (current_time - performance_stats['last_attack_time'] < 10):
            attack = performance_stats['last_attack']
            print(f" | Last: {attack['type']} ({attack['src']} -> {attack['dst']})", end='')
        
        # Print packet details for first few packets
        if performance_stats['total_packets'] <= 5:
            print("\n[DEBUG] Packet details:")
            try:
                if pkt.haslayer('IP'):
                    ip = pkt['IP']
                    proto = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}.get(ip.proto, f'Proto({ip.proto})')
                    print(f"    {ip.src}:{getattr(pkt, 'sport', '?')} -> {ip.dst}:{getattr(pkt, 'dport', '?')} {proto}")
                    print(f"    Length: {len(pkt)} bytes")
                    
            except Exception as e:
                print(f"    Error: {str(e)}")
        
        print('\n' + '.' * (performance_stats['total_packets'] // 10 % 100), end='', flush=True)
=======
    performance_stats['last_packet_time'] = time.time()

    # Simplified activity indicator (only every 100 packets)
    current_time = time.time()
    if performance_stats['total_packets'] % 100 == 0:
        print('.', end='', flush=True)

    # Debug: Print packet info only for first few packets
    if performance_stats['total_packets'] <= 3:
        print(f"\n[DEBUG] Packet #{performance_stats['total_packets']}:")

        try:
            # Basic packet info
            print(f"    Time: {time.strftime('%H:%M:%S', time.localtime(current_time))}")
            print(f"    Type: {type(pkt).__name__}")

            # IP layer info
            if pkt.haslayer('IP'):
                ip = pkt['IP']
                print(f"    IP: {ip.src} -> {ip.dst} proto={ip.proto}")
            elif pkt.haslayer('IPv6'):
                ipv6 = pkt['IPv6']
                print(f"    IPv6: {ipv6.src} -> {ipv6.dst} nh={ipv6.nh}")

            print(f"    Length: {len(pkt)} bytes")

        except Exception as e:
            print(f"    Error getting packet details: {str(e)}")
>>>>>>> org/master
    
    try:
        # Start processing time
        start_time = time.time()
        
        # Check if packet has IP layer
        has_ip = pkt.haslayer('IP')
        has_ipv6 = pkt.haslayer('IPv6')
        
        # Initialize feature dictionary
        features = {}
        
        # Extract basic packet info
        if has_ip or has_ipv6:
            ip = pkt['IP'] if has_ip else pkt['IPv6']
            proto = ip.proto if has_ip else ip.nh
            
            # Basic features
            features.update({
                'src_ip': ip.src,
                'dst_ip': ip.dst,
                'protocol': proto,
                'length': len(pkt),
                'timestamp': current_time
            })
            
            # Ports if available
            for port in ['sport', 'dport']:
                if hasattr(pkt, port):
                    features[port] = getattr(pkt, port)
            
            # TCP flags if available
            if pkt.haslayer('TCP'):
                tcp = pkt['TCP']
                features.update({
                    'tcp_flags': tcp.flags,
                    'tcp_window': tcp.window,
                    'tcp_ack': tcp.ack if hasattr(tcp, 'ack') else 0,
                    'tcp_seq': tcp.seq if hasattr(tcp, 'seq') else 0
                })
            
            # Make prediction
            if use_cic_model and 'model' in globals():
                try:
                    # Map features to CIC format
                    cic_features = map_pcap_features_to_cic(features)
                    
                    # Make prediction
                    prediction, confidence = predict_with_cic_model(cic_features)
                    
                    # Update stats
                    if prediction != 'Benign' and prediction != 'Normal':
                        performance_stats['attack_detected'] += 1
                        attack_info = {
                            'type': prediction,
                            'confidence': f"{confidence*100:.1f}%",
                            'src': f"{features.get('src_ip', '?')}:{features.get('sport', '?')}",
                            'dst': f"{features.get('dst_ip', '?')}:{features.get('dport', '?')}",
                            'time': time.strftime('%H:%M:%S')
                        }
                        performance_stats['last_attack'] = attack_info
                        performance_stats['last_attack_time'] = current_time
                        
                        # Print attack alert
                        print(f"\n\n{'!'*80}\n"
                              f"🚨 ATTACK DETECTED: {prediction} (Confidence: {confidence*100:.1f}%)\n"
                              f"   Source: {attack_info['src']} -> Destination: {attack_info['dst']}\n"
                              f"   Time: {attack_info['time']}\n"
                              f"{'!'*80}\n")
                    
                    # Update prediction stats
                    if prediction not in performance_stats['predictions']:
                        performance_stats['predictions'][prediction] = 0
                    performance_stats['predictions'][prediction] += 1
                    
                    # Periodic prediction summary
                    if current_time - performance_stats.get('last_prediction_time', 0) > 30:  # Every 30 seconds
                        print("\n" + "="*60)
                        print("PREDICTION SUMMARY (last 30s):")
                        for pred, count in performance_stats['predictions'].items():
                            print(f"  - {pred}: {count}")
                        print("="*60 + "\n")
                        performance_stats['predictions'] = {}
                        performance_stats['last_prediction_time'] = current_time
                    
                except Exception as e:
                    if 'prediction_errors' not in performance_stats:
                        performance_stats['prediction_errors'] = 0
                    performance_stats['prediction_errors'] += 1
                    if performance_stats['prediction_errors'] <= 3:  # Only show first few errors
                        print(f"\n[!] Prediction error: {str(e)}")
        
        performance_stats['processed_packets'] += 1
        
        if not (has_ip or has_ipv6):
            performance_stats['dropped_packets'] = performance_stats.get('dropped_packets', 0) + 1
            if performance_stats['total_packets'] % 50 == 0:  # Less frequent non-IP packet logging
                print(f"\n[!] Non-IP packet dropped (total: {performance_stats['dropped_packets']})")
                try:
                    print(f"    Packet layers: {[l.name for l in pkt.layers() if hasattr(l, 'name')]}")
                except:
                    print("    Could not get layer information")
            return True
            
        # Extract basic packet info with better error handling
        try:
            if has_ip:
                ip_layer = pkt['IP']
                src_ip = ip_layer.src
                dst_ip = ip_layer.dst
                proto = ip_layer.proto
                sport = pkt.sport if hasattr(pkt, 'sport') else 0
                dport = pkt.dport if hasattr(pkt, 'dport') else 0
                ttl = ip_layer.ttl if hasattr(ip_layer, 'ttl') else 0
            else:  # IPv6
                ip_layer = pkt['IPv6']
                src_ip = ip_layer.src
                dst_ip = ip_layer.dst
                proto = ip_layer.nh
                sport = pkt.sport if hasattr(pkt, 'sport') else 0
                dport = pkt.dport if hasattr(pkt, 'dport') else 0
                ttl = ip_layer.hlim if hasattr(ip_layer, 'hlim') else 0
                
            # Log periodic summary (every 500 packets)
            if performance_stats['total_packets'] % 500 == 0:
                elapsed = time.time() - performance_stats.get('last_summary', current_time)
                pps = 500 / elapsed if elapsed > 0 else 0
                print(f"\n[+] Packets: {performance_stats['total_packets']} | "
                      f"Processed: {performance_stats.get('processed_packets', 0)} | "
                      f"Dropped: {performance_stats.get('dropped_packets', 0)} | "
                      f"Rate: {pps:.1f} pps")
                performance_stats['last_summary'] = current_time
                
        except Exception as e:
            print(f"\n[!] Error extracting packet info: {str(e)}")
            performance_stats['dropped_packets'] = performance_stats.get('dropped_packets', 0) + 1
            return True
        
        # Reduced packet info logging (every 200 packets)
        if performance_stats['total_packets'] % 200 == 0:
            print(f"\n[+] Packets processed: {performance_stats['total_packets']}")
            print(f"    Last packet: {src_ip}:{sport} -> {dst_ip}:{dport} proto={proto}")
        
        # Create a cache key for this flow
        cache_key = (src_ip, dst_ip, proto, sport, dport)
        
        # Check cache for previous decision (fast path for known benign traffic)
        if cache_key in DECISION_CACHE:
            decision = DECISION_CACHE[cache_key]
            if decision == 'benign':
                performance_stats['dropped_packets'] += 1
                update_performance_stats(False)
                return True
        
        # Fast path: Skip common benign traffic
        if is_benign_port(dport) or is_benign_port(sport):
            # Update cache with benign decision
            if len(DECISION_CACHE) < CACHE_SIZE:
                DECISION_CACHE[cache_key] = 'benign'
            performance_stats['dropped_packets'] += 1
            update_performance_stats(False)
            return True
        
        # Check for port scanning patterns (optimized)
        if dport > 0 and is_port_scan_enhanced(pkt):
            scan_alert = is_port_scan_enhanced(pkt)
            if scan_alert:
                print(f"\n\033[1;31m🚨 {scan_alert} 🚨\033[0m")
                print(f"  Source: \033[94m{src_ip}\033[0m")
                print(f"  Target: \033[94m{dst_ip}:{dport}\033[0m")
                print(f"  Protocol: \033[96m{proto}\033[0m")
                print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("\033[90m" + "-" * 60 + "\033[0m")
                
                # Update cache with malicious decision
                if len(DECISION_CACHE) < CACHE_SIZE:
                    DECISION_CACHE[cache_key] = 'malicious'
                update_performance_stats(True)
                return True
        
        # Generate flow ID for feature extraction
        flow_id = f"{src_ip}-{dst_ip}-{proto}-{sport}-{dport}"

        # Extract features using enhanced feature extractor
        try:
            if ENHANCED_DETECTION_AVAILABLE and extract_flow_features:
                features = extract_flow_features(pkt, flow_tracker, flow_id)
                if not isinstance(features, dict):
                    print(f"\n[!] Invalid features format: {type(features)}")
                    return True
            else:
                # Fallback to basic feature extraction
                features = {
                    'length': len(pkt),
                    'src_bytes': len(pkt),
                    'packet_count': 1,
                    'flow_duration': 0.1
                }
        except Exception as e:
            print(f"\n[!] Error in extract_flow_features: {e}")
            return True
        
        # Add basic packet info to features
        features.update({
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'protocol': proto,
            'sport': sport,
            'dport': dport,
            'timestamp': time.time()
        })
        
        # Add packet to processing queue for advanced detection if available
        try:
            if ENHANCED_DETECTION_AVAILABLE and packet_processor is not None:
                packet_processor.add_packet(pkt, features)
        except Exception as e:
            if int(time.time()) % 10 == 0:  # Log occasionally
                print(f"\n[!] Error in packet_processor: {e}")
            return True
        
        # Update performance stats safely
        performance_stats['processed_packets'] = performance_stats.get('processed_packets', 0) + 1
        update_performance_stats(True)
        
        # For logging and legacy compatibility
        protocol = 'TCP' if proto == 6 else 'UDP' if proto == 17 else 'ICMP' if proto == 1 else str(proto)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pkt_len = len(pkt)
        
        try:
            # Get prediction based on model type
            start_time = time.time()
            try:
                if use_cic_model:
                    prediction_label, confidence = predict_with_cic_model(features)
                else:
                    prediction_label = predict_with_pcap_model(features, pcap_model, pcap_encoder)
                    confidence = 1.0  # Default confidence for pcap model

                # Debug output for predictions (reduced frequency)
                if performance_stats['total_packets'] % 50 == 0:  # Log every 50th packet
                    attack_name = get_attack_name(prediction_label)
                    print(f"\n[+] Model Prediction:")
                    print(f"    Label: {prediction_label}")
                    print(f"    Attack Type: {attack_name}")
                    print(f"    Confidence: {confidence*100:.2f}%")
                    print(f"    Flow: {src_ip}:{sport} -> {dst_ip}:{dport} {proto}")
                
                # Check if this is likely a false positive
                is_benign = False
                prediction_str = str(prediction_label).lower()
                
                # Check for explicitly benign/unknown labels
                if prediction_str in ['benign', 'normal', '0', 'prediction_error', 'error', 'unknown']:
                    is_benign = True
                elif prediction_str.isdigit() and int(prediction_str) == 0:
                    is_benign = True
                # Check for common web traffic false positives
                elif is_benign_traffic(prediction_label, sport, dport, 'TCP' if proto == 6 else 'UDP' if proto == 17 else str(proto)):
                    is_benign = True
                
                # Only alert on high-confidence attacks
                if not is_benign and confidence > 0.7:  # 70% confidence threshold
                    attack_name = get_attack_name(prediction_label)
                    print(f"\n\033[1;31m🚨 POTENTIAL ATTACK DETECTED 🚨\033[0m")
                    print(f"  Type: {attack_name} (Confidence: {confidence*100:.1f}%)")
                    print(f"  From: {src_ip}:{sport} -> {dst_ip}:{dport} {proto}")
                    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print("\033[90m" + "-" * 60 + "\033[0m")
                    
                    # Update attack counter
                    performance_stats['attack_detected'] = performance_stats.get('attack_detected', 0) + 1
                    
                    # Update cache with malicious decision
                    if len(DECISION_CACHE) < CACHE_SIZE:
                        DECISION_CACHE[cache_key] = 'malicious'
                
                prediction_time = (time.time() - start_time) * 1000  # in ms
                
                # Log to debug file
                if 'DEBUG_LOG_FILE' in globals():
                    try:
                        with open(DEBUG_LOG_FILE, 'a') as f:
                            f.write(f"{datetime.now().isoformat()},{src_ip},{sport},{dst_ip},{dport},{proto},{prediction_label},{confidence}\n")
                    except Exception as e:
                        print(f"[!] Error writing to debug log: {e}")
                        
            except Exception as e:
                print(f"\n[!] Error during prediction: {e}")
                performance_stats['errors'] = performance_stats.get('errors', 0) + 1
                return True
            
            # Log for debugging
            log_debug_info(features, prediction_label)
            
            # PostgreSQL-specific false positive handling
            if str(prediction_label) == '28' and not is_benign:  # Only check if not already marked as benign
                # If this doesn't look like PostgreSQL traffic, mark as benign
                if dport != 5432 and not (3306 <= dport <= 3400):  # Common DB ports
                    is_benign = True
            
            if not is_benign:
                # Get attack name
                attack_name = get_attack_name(prediction_label)
                
                # Format confidence color and adjust threshold for certain attack types
                conf_color = "\033[91m"  # Red for high confidence
                confidence_threshold = 0.85  # Slightly higher default threshold
                
                # Adjust threshold for certain attack types
                if str(prediction_label) in ['28']:  # PostgreSQL
                    confidence_threshold = 0.98  # Very high confidence for PostgreSQL
                
                # Skip if confidence is below threshold
                if confidence < confidence_threshold:
                    return True
                
                # Set color based on confidence
                if confidence < 0.9:
                    conf_color = "\033[93m"  # Yellow for medium confidence
                elif confidence < 0.85:
                    conf_color = "\033[92m"  # Green for low confidence
                
                # Print alert
                print(f"\n\033[1;31m🚨 Suspicious activity detected! 🚨\033[0m")
                print(f"  Type: {attack_name} (ID: {prediction_label})")
                print(f"  Confidence: {conf_color}{confidence*100:.1f}%\033[0m")
                print(f"  Source: \033[94m{src_ip}:{sport}\033[0m")
                print(f"  Destination: \033[94m{dst_ip}:{dport}\033[0m")
                print(f"  Protocol: \033[96m{protocol}\033[0m")
                print(f"  Length: {pkt_len:,} bytes")
                print(f"  Prediction time: {prediction_time:.2f}ms")
                print(f"  Timestamp: {timestamp}")
                print("\033[90m" + "-" * 60 + "\033[0m")
            
            # Update label counter
            if prediction_label not in label_counter:
                label_counter[prediction_label] = 0
            label_counter[prediction_label] += 1
            
            # Log summary periodically
            total_packets = sum(label_counter.values())
            if total_packets > 0 and total_packets % 20 == 0:  # Less frequent updates
                # Create a summary with attack names, sorted by count
                summary = {}
                for label, count in label_counter.items():
                    attack_name = get_attack_name(label)
                    summary[attack_name] = summary.get(attack_name, 0) + count
                
                # Sort by count (descending)
                sorted_summary = dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))
                
                # Print summary
                print("\n" + "="*50)
                print(f"📊 Detection Summary (Total packets: {total_packets}):")
                for attack, count in sorted_summary.items():
                    print(f"  {attack}: {count}")
                print("="*50 + "\n")
                
                # Also log to file
                logging.info(f"Packet Summary (Total: {total_packets}): {sorted_summary}")
            
            return True
                
        except Exception as e:
            error_msg = f"Error processing packet: {str(e)}"
            logging.error(error_msg)
            print(error_msg)
            return True
        
    except Exception as e:
        error_msg = f"Error in packet_handler: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        return True


def start_sniffing(interface=DEFAULT_INTERFACE, packet_count=0, bpf_filter="ip or ip6",
                  use_cic_model=True, pcap_model=None, pcap_encoder=None):
    """Start optimized packet sniffing with performance monitoring"""
    global connection_tracker, performance_stats
    
    # Initialize connection tracker
    connection_tracker = FastConnectionTracker()
    
    print("\n" + "="*70)
    print(f"{'Network Intrusion Detection System':^70}")
    print(f"{'High-Performance Mode':^70}")
    print("="*70)
    print(f"\n[+] Starting sniffer on interface: {interface}")
    print(f"[+] BPF Filter: {bpf_filter}")
    print(f"[+] Using {'CIC-trained' if use_cic_model else 'PCAP-trained'} model")
    print(f"[+] Cache size: {CACHE_SIZE} flows")
    print("\n[!] Press Ctrl+C to stop\n")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILENAME),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Optimize system settings if possible
        try:
            import socket
            sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)  # 8MB buffer
            sock.close()
        except Exception as e:
            print(f"[!] Warning: Could not optimize socket settings: {e}")
        
        # Verify interface is up and has an IP address
        try:
            import netifaces
            if interface not in netifaces.interfaces():
                print(f"[!] Error: Interface {interface} not found!")
                available = netifaces.interfaces()
                print(f"[+] Available interfaces: {', '.join(available) if available else 'None'}")
                return
                
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET not in addrs:
                print(f"[!] Warning: Interface {interface} has no IPv4 address")
                print("[*] Trying to continue anyway...")
                
        except ImportError:
            print("[*] netifaces module not available, skipping interface verification")
        
        # Configure Scapy for better packet capture
        conf.use_pcap = True  # Use libpcap if available
        conf.iface = interface
        
        # Print debug info
        print(f"[+] Using interface: {conf.iface}")
        print(f"[+] BPF filter: {bpf_filter}")
        print("[*] Starting packet capture...")
        
        # Start sniffing with continuous capture
        print("\n[!] Starting packet capture (press Ctrl+C to stop)...")
        try:
            # Use a simple while loop to keep sniffing
            sniff(
                iface=interface,
                prn=lambda pkt: packet_handler(pkt, use_cic_model, pcap_model, pcap_encoder),
                store=0,  # Don't store packets in memory
                filter=bpf_filter,
                count=0,  # Capture indefinitely
                timeout=None,  # No timeout
                promisc=True,  # Promiscuous mode
                stop_filter=lambda x: False  # Never stop unless interrupted
            )
        except Exception as e:
            print(f"\n[!] Error starting sniffer: {e}")
            # Try with a simpler configuration
            print("[*] Trying alternative sniffing method...")
            sniff(
                iface=interface,
                prn=lambda pkt: packet_handler(pkt, use_cic_model, pcap_model, pcap_encoder),
                store=0,
                filter=bpf_filter,
                count=packet_count
            )
        
    except KeyboardInterrupt:
        print("\n\n[!] Sniffer stopped by user")
    except Exception as e:
        logging.critical(f"Sniffing failed: {e}")
    finally:
        # Print final statistics
        total = performance_stats['total_packets'] + performance_stats['processed_packets'] + performance_stats['dropped_packets']
        processed = performance_stats['processed_packets']
        dropped = performance_stats['dropped_packets']
        drop_rate = (dropped / total * 100) if total > 0 else 0
        
        print("\n" + "="*50)
        print("Sniffing Session Summary:")
        print("="*50)
        print(f"Total packets: {total}")
        print(f"Processed: {processed}")
        print(f"Dropped: {dropped} ({drop_rate:.2f}%)")
        print(f"Active connections: {len(connection_tracker.connections) if connection_tracker else 0}")
        print(f"Cache size: {len(DECISION_CACHE)}")
        print("="*50 + "\n")
        
        logging.info("Sniffing session ended.")
        logging.info(f"Final summary: {dict(label_counter)}")


def main():
    parser = argparse.ArgumentParser(
        description="High-Performance Network Intrusion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Basic usage: python %(prog)s -i eth0
  With custom model: %(prog)s -i eth0 --model-path models/custom_model.pkl
  With specific filter: %(prog)s -i eth0 -f "tcp port 80 or tcp port 443"
"""
    )
    
    # Sniffing options
    sniff_group = parser.add_argument_group('Sniffing Options')
    sniff_group.add_argument("-i", "--interface", type=str, default=DEFAULT_INTERFACE,
                           help=f"Network interface (default: {DEFAULT_INTERFACE})")
    sniff_group.add_argument("-c", "--count", type=int, default=0,
                           help="Number of packets to sniff (0 = infinite, default: 0)")
    sniff_group.add_argument("-f", "--filter", type=str, default="ip or ip6",
                           help="BPF filter string (default: ip or ip6)")
    sniff_group.add_argument("--cache-size", type=int, default=10000,
                           help=f"Flow cache size (default: 10000)")
    
    # Model options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument("--model-type", type=str, choices=["cic", "pcap"], default="cic",
                           help="Type of model to use (default: cic)")
    model_group.add_argument("--model-path", type=str,
                           default="models/cic_models/random_forest_model.pkl",
                           help="Path to model file")
    model_group.add_argument("--encoder-path", type=str,
                           default="artifacts/cic_label_encoder.pkl",
                           help="Path to label encoder")
    model_group.add_argument("--scaler-path", type=str,
                           default="artifacts/scaler.joblib",
                           help="Path to feature scaler (default: artifacts/scaler.joblib)")
    model_group.add_argument("--features-path", type=str,
                           default="artifacts/selected_features.txt",
                           help="Path to selected features list")
    
    # Performance options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument("--no-ml", action="store_true",
                          help="Disable ML-based detection (use only rule-based)")
    perf_group.add_argument("--rate-limit", type=int, default=5000,
                          help="Maximum packets per second per IP (default: 5000)")
    
    args = parser.parse_args()
    
    # Update global cache size
    global CACHE_SIZE
    CACHE_SIZE = args.cache_size
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILENAME, mode='a'),
            logging.StreamHandler()
        ]
    )
    
    # Set console logging format
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', "%H:%M:%S")
    console.setFormatter(formatter)
    
    # Remove any existing console handlers
    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            logging.root.removeHandler(handler)
    
    # Add our console handler
    logging.getLogger('').addHandler(console)
    
    try:
        # Load model if ML is enabled
        if not args.no_ml:
            # Ensure the artifacts directory exists
            os.makedirs("artifacts", exist_ok=True)
            
            # If using default paths, check if the files exist and provide guidance if not
            if args.model_type == "cic" and args.model_path == "models/cic_models/random_forest_model.pkl" and not os.path.exists(args.model_path):
                print("\n⚠️  Default model not found. Please train a model first using train_cic_pipeline.py")
                print("   or specify a custom model path with --model-path")
                print("\nTo train a model, run:")
                print("  python src/train_cic_pipeline.py --cic-path path/to/your/cic/dataset")
                sys.exit(1)
                
            if args.model_type == "cic":
                load_cic_model(
                    model_path=args.model_path,
                    encoder_path=args.encoder_path,
                    scaler_path=args.scaler_path,
                    features_path=args.features_path
                )
                use_cic_model = True
                pcap_model = None
                pcap_encoder = None
            else:
                # Load PCAP model
                pcap_model = joblib.load("models/xgboost_model_final.pkl")
                pcap_encoder = joblib.load("models/label_encoder_xgboost.pkl")
                use_cic_model = False
        else:
            logging.info("ML-based detection is disabled (running in rule-based mode only)")
            use_cic_model = False
            pcap_model = None
            pcap_encoder = None
        
        # Update rate limiter
        global rate_limiter
        rate_limiter = RateLimiter(max_events=args.rate_limit, time_window=1)  # Per second
        
        # Start sniffing
        start_sniffing(
            interface=args.interface,
            packet_count=args.count,
            bpf_filter=args.filter,
            use_cic_model=use_cic_model,
            pcap_model=pcap_model,
            pcap_encoder=pcap_encoder
        )
        
    except KeyboardInterrupt:
        print("\n[!] Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        if 'debug' in globals() and debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
