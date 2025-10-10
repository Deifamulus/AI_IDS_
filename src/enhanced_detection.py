"""
Enhanced attack detection components for the network sniffer
"""
import math
import string
import logging
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import time
from typing import Dict, List, Tuple, Optional, Any

class AdvancedAttackDetector:
    """Enhanced attack detection with multiple detection methods"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.flow_states = defaultdict(dict)
        self.scan_detector = PortScanDetector()
        self.dos_detector = DoSDetector()
        self.anomaly_detector = AnomalyDetector()
        self.ml_detector = None
        self.logger = logging.getLogger('detection')
        
        # Initialize ML detector if model paths are provided
        if all(k in self.config for k in ['model_path', 'scaler_path', 'feature_columns']):
            try:
                self.ml_detector = MLDetector(
                    model_path=self.config['model_path'],
                    scaler_path=self.config['scaler_path'],
                    feature_columns=self.config['feature_columns']
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize ML detector: {e}")
    
    def detect_attacks(self, pkt: Any, features: Dict) -> List[str]:
        """
        Detect various types of attacks using multiple detection methods
        
        Args:
            pkt: The raw packet
            features: Extracted packet features
            
        Returns:
            List of detected attack types
        """
        results = []
        
        try:
            # 1. Port scan detection
            scan_result = self.scan_detector.analyze(pkt, features)
            if scan_result:
                results.append(scan_result)
            
            # 2. DoS/DDoS detection
            dos_result = self.dos_detector.analyze(pkt, features)
            if dos_result:
                results.append(dos_result)
            
            # 3. Protocol anomaly detection
            if pkt.haslayer('TCP'):
                anomaly = self.anomaly_detector.detect_tcp_anomalies(pkt, features)
                if anomaly:
                    results.append(anomaly)
            
            # 4. Payload analysis
            if pkt.haslayer('Raw'):
                payload = str(pkt['Raw'].load)
                if self._detect_shellcode(payload):
                    results.append("SHELLCODE_DETECTED")
            
            # 5. ML-based detection
            if self.ml_detector:
                ml_result = self.ml_detector.detect(features)
                if ml_result and ml_result[1] > 0.8:  # Confidence threshold
                    results.append(f"ML_DETECTION_{ml_result[0]}")
            
            return list(set(results))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error in attack detection: {e}")
            return []
    
    def _detect_shellcode(self, payload: str) -> bool:
        """Detect potential shellcode in payload"""
        if not payload:
            return False
            
        # Simple NOP sled detection
        nop_sequences = ["\x90" * 8, "\x0f\x1f\x40\x00"]  # Common NOP sequences
        if any(seq in payload for seq in nop_sequences):
            return True
            
        # Detect high entropy which might indicate encrypted/obfuscated code
        entropy = self._calculate_entropy(payload)
        if entropy > 6.5:  # High entropy threshold
            return True
            
        return False
    
    @staticmethod
    def _calculate_entropy(data: str) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
            
        entropy = 0.0
        for x in range(256):
            p_x = float(data.count(chr(x))) / len(data)
            if p_x > 0:
                entropy += -p_x * math.log2(p_x)
        return entropy


class PortScanDetector:
    """Detect various types of port scans"""
    
    def __init__(self, window_size: int = 60, threshold: int = 5):
        self.window_size = window_size  # seconds
        self.threshold = threshold     # min connections to trigger
        self.scan_attempts = defaultdict(lambda: {
            'ports': set(),
            'targets': set(),
            'timestamps': deque(maxlen=1000)
        })
        self.lock = threading.Lock()
    
    def analyze(self, pkt: Any, features: Dict) -> Optional[str]:
        """Analyze packet for port scan patterns"""
        if not all(k in features for k in ['src_ip', 'dst_ip', 'dport']):
            return None
            
        src_ip = features['src_ip']
        dst_ip = features['dst_ip']
        dport = features['dport']
        current_time = time.time()
        
        with self.lock:
            # Clean old entries
            self._cleanup_old_entries(current_time)
            
            # Update scan attempts
            if src_ip not in self.scan_attempts:
                self.scan_attempts[src_ip] = {
                    'ports': set(),
                    'targets': set(),
                    'timestamps': deque(maxlen=1000)
                }
            
            scan_data = self.scan_attempts[src_ip]
            scan_data['ports'].add(dport)
            scan_data['targets'].add(dst_ip)
            scan_data['timestamps'].append(current_time)
            
            # Check for horizontal scan (multiple ports on single host)
            if len(scan_data['ports']) >= self.threshold and len(scan_data['targets']) == 1:
                return f"HORIZONTAL_SCAN from {src_ip} to {dst_ip} ({len(scan_data['ports'])} ports)"
            
            # Check for vertical scan (single port on multiple hosts)
            if len(scan_data['targets']) >= self.threshold and len(scan_data['ports']) == 1:
                return f"VERTICAL_SCAN from {src_ip} (port {dport} to {len(scan_data['targets'])} hosts)"
            
            # Check for distributed scan (multiple ports on multiple hosts)
            if (len(scan_data['targets']) >= self.threshold and 
                len(scan_data['ports']) >= self.threshold):
                return f"DISTRIBUTED_SCAN from {src_ip} ({len(scan_data['ports'])} ports to {len(scan_data['targets'])} hosts)"
            
            return None
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove entries older than window_size"""
        to_remove = []
        
        for src_ip, data in self.scan_attempts.items():
            # Remove old timestamps
            while (data['timestamps'] and 
                   current_time - data['timestamps'][0] > self.window_size):
                data['timestamps'].popleft()
            
            # If no recent activity, mark for removal
            if not data['timestamps']:
                to_remove.append(src_ip)
        
        # Remove inactive entries
        for src_ip in to_remove:
            self.scan_attempts.pop(src_ip, None)


class DoSDetector:
    """Detect DoS/DDoS attacks"""
    
    def __init__(self, window_size: int = 10, threshold: float = 1000):
        self.window_size = window_size  # seconds
        self.threshold = threshold     # packets/second threshold
        self.packet_counts = defaultdict(int)
        self.timestamps = deque()
        self.lock = threading.Lock()
    
    def analyze(self, pkt: Any, features: Dict) -> Optional[str]:
        """Analyze packet for DoS patterns"""
        if not pkt.haslayer('IP'):
            return None
            
        current_time = time.time()
        
        with self.lock:
            # Add current timestamp
            self.timestamps.append(current_time)
            
            # Remove timestamps outside the window
            while self.timestamps and (current_time - self.timestamps[0]) > self.window_size:
                self.timestamps.popleft()
            
            # Calculate packets per second
            pps = len(self.timestamps) / self.window_size
            
            if pps > self.threshold:
                return f"POTENTIAL_DOS_DETECTED: {pps:.1f} packets/second"
            
            return None


class AnomalyDetector:
    """Detect protocol anomalies"""
    
    def detect_tcp_anomalies(self, pkt: Any, features: Dict) -> Optional[str]:
        """Detect TCP protocol anomalies"""
        if not pkt.haslayer('TCP'):
            return None
            
        tcp = pkt['TCP']
        flags = str(tcp.flags)
        
        # TCP flag anomalies
        if 'S' in flags and 'A' in flags and 'F' in flags:  # SYN-ACK-FIN
            return "TCP_ANOMALY: SYN-ACK-FIN flags set"
            
        if 'S' in flags and 'R' in flags:  # SYN-RST
            return "TCP_ANOMALY: SYN-RST flags set"
            
        # Window size anomalies
        if tcp.window == 0 and 'A' not in flags:  # Zero window without ACK
            return "TCP_ANOMALY: Zero window without ACK"
            
        return None


class MLDetector:
    """ML-based attack detection"""
    
    def __init__(self, model_path: str, scaler_path: str, feature_columns: List[str]):
        import joblib
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = feature_columns
        self.cache = {}
        self.cache_size = 10000
        self.cache_keys = deque()
    
    def detect(self, features: Dict) -> Tuple[Any, float]:
        """Detect anomalies using ML model"""
        try:
            # Create cache key
            cache_key = tuple(sorted((k, str(v)) for k, v in features.items()))
            
            # Check cache
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Prepare features
            X = self._prepare_features(features)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X).max()
            
            # Update cache
            self._update_cache(cache_key, (prediction, probability))
            
            return prediction, probability
            
        except Exception as e:
            logging.error(f"ML detection error: {e}")
            return None, 0.0
    
    def _prepare_features(self, features: Dict) -> np.ndarray:
        """Prepare features for ML model"""
        # Create DataFrame with expected columns
        df = pd.DataFrame([features])
        
        # Ensure all expected columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select and scale features
        X = df[self.feature_columns]
        return self.scaler.transform(X)
    
    def _update_cache(self, key: Any, value: Any):
        """Update prediction cache with LRU policy"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            old_key = self.cache_keys.popleft()
            self.cache.pop(old_key, None)
        
        # Add new item
        self.cache[key] = value
        self.cache_keys.append(key)


class OptimizedPacketProcessor:
    """High-performance packet processing with batching"""
    
    def __init__(self, detector: AdvancedAttackDetector, batch_size: int = 100, max_workers: int = 4):
        self.detector = detector
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.packet_queue = Queue(maxsize=1000)
        self.running = False
        self.worker_thread = None
        self.logger = logging.getLogger('processor')
    
    def start(self):
        """Start the packet processing loop"""
        if self.running:
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.worker_thread.start()
        self.logger.info("Packet processor started")
    
    def stop(self):
        """Stop the packet processing loop"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.executor.shutdown(wait=False)
        self.logger.info("Packet processor stopped")
    
    def add_packet(self, pkt: Any, features: Dict) -> bool:
        """Add packet to processing queue"""
        try:
            self.packet_queue.put((pkt, features), block=False)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to add packet to queue: {e}")
            return False
    
    def _process_batches(self):
        """Process packets in batches for better performance"""
        batch = []
        last_process_time = time.time()
        
        while self.running:
            try:
                # Add packet to batch with timeout
                pkt, features = self.packet_queue.get(timeout=1)
                batch.append((pkt, features))
                
                # Process batch if size reached or timeout
                current_time = time.time()
                if (len(batch) >= self.batch_size or 
                    (current_time - last_process_time) >= 0.5):  # Max 500ms delay
                    self._process_batch(batch)
                    batch = []
                    last_process_time = current_time
                    
            except Exception as e:
                self.logger.error(f"Error in batch processing: {e}")
                # Process any remaining packets in batch
                if batch:
                    self._process_batch(batch)
                    batch = []
                    last_process_time = time.time()
    
    def _process_batch(self, batch: List[Tuple[Any, Dict]]):
        """Process a batch of packets in parallel"""
        if not batch:
            return
            
        futures = []
        for pkt, features in batch:
            future = self.executor.submit(self._process_single, pkt, features)
            futures.append(future)
        
        # Wait for all futures to complete with timeout
        for future in futures:
            try:
                future.result(timeout=1)  # 1 second timeout per packet
            except Exception as e:
                self.logger.error(f"Error processing packet: {e}")
    
    def _process_single(self, pkt: Any, features: Dict):
        """Process a single packet"""
        try:
            # Detect attacks
            attacks = self.detector.detect_attacks(pkt, features)
            
            # Log detected attacks
            if attacks:
                src_ip = features.get('src_ip', 'unknown')
                dst_ip = features.get('dst_ip', 'unknown')
                self.logger.warning(
                    f"Detected attacks from {src_ip} to {dst_ip}: {', '.join(attacks)}"
                )
                
        except Exception as e:
            self.logger.error(f"Error in single packet processing: {e}")


def extract_flow_features(pkt: Any, flow_tracker: Any, flow_id: str) -> Dict:
    """
    Extract enhanced flow-based features from packet
    
    Args:
        pkt: The raw packet
        flow_tracker: Flow tracker instance
        flow_id: Unique flow identifier
        
    Returns:
        Dictionary of extracted features
    """
    features = {}
    
    try:
        # 1. Basic packet information
        if pkt.haslayer('IP'):
            ip = pkt['IP']
            features['src_ip'] = ip.src
            features['dst_ip'] = ip.dst
            features['ip_len'] = ip.len
            features['ip_ttl'] = ip.ttl
            features['ip_proto'] = ip.proto
        
        # 2. TCP features
        if pkt.haslayer('TCP'):
            tcp = pkt['TCP']
            features['sport'] = tcp.sport
            features['dport'] = tcp.dport
            features['tcp_flags'] = str(tcp.flags)
            features['tcp_window'] = tcp.window
            features['tcp_seq'] = tcp.seq
            features['tcp_ack'] = tcp.ack
        
        # 3. UDP features
        elif pkt.haslayer('UDP'):
            udp = pkt['UDP']
            features['sport'] = udp.sport
            features['dport'] = udp.dport
            features['udp_len'] = udp.len
        
        # 4. Time-based features
        current_time = time.time()
        if flow_id in flow_tracker:
            flow = flow_tracker[flow_id]
            flow_duration = current_time - flow.get('start_time', current_time)
            features['flow_duration'] = flow_duration
            features['packets_per_second'] = flow.get('packet_count', 1) / max(flow_duration, 0.001)
        else:
            features['flow_duration'] = 0
            features['packets_per_second'] = 0
        
        # 5. Payload features
        if pkt.haslayer('Raw'):
            payload = str(pkt['Raw'].load)
            features['payload_length'] = len(payload)
            features['entropy'] = calculate_entropy(payload)
            features['printable_ratio'] = calculate_printable_ratio(payload)
        
        return features
        
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return {}


def calculate_entropy(data: str) -> float:
    """Calculate Shannon entropy of data"""
    if not data:
        return 0.0
        
    entropy = 0.0
    for x in range(256):
        p_x = float(data.count(chr(x))) / len(data)
        if p_x > 0:
            entropy += -p_x * math.log2(p_x)
    return entropy


def calculate_printable_ratio(data: str) -> float:
    """Calculate ratio of printable characters in data"""
    if not data:
        return 0.0
        
    printable = sum(c in string.printable for c in data)
    return printable / len(data) if data else 0.0
