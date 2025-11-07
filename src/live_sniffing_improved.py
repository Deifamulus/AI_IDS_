
#!/usr/bin/env python3
"""
Suricata-based Network IDS
A lightweight network intrusion detection system using Suricata for traffic analysis.
"""

import os
import sys
import time
import logging
import argparse
import threading
import ipaddress
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Deque
from pathlib import Path
import subprocess
import json
from collections import deque
from dataclasses import dataclass
import requests
import socketio as socketio_client

import scapy.all as scapy

# Dashboard settings
DASHBOARD_URL = "http://localhost:5000/update"

# -------------------------
# Configuration & Constants
# -------------------------
DEFAULT_INTERFACE = "lo"
LOG_FILENAME = "suricata_ids.log"
DEFAULT_RULES_DIR = "/var/lib/suricata/rules"
EVE_JSON_PATH = "/var/log/suricata/eve.json"

# Alert parameters
SURICATA_ALERT_WINDOW = 30  # seconds to consider a recent suricata alert relevant

# Whitelists and false-positive suppression (minimal defaults)
WHITELISTED_PORTS = {80, 443, 53, 123}
# Only whitelist localhost by default
WHITELISTED_IP_RANGES = {
    "127.0.0.1/32",
    "::1/128"  # IPv6 localhost
}

# -------------------------
# Utilities
# -------------------------
def setup_logging(debug: bool = True) -> logging.Logger:  # Changed default to True
    logger = logging.getLogger("nids_improved_fusion")
    logger.setLevel(logging.DEBUG)  # Always set to DEBUG
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # Clear existing handlers
    if logger.handlers:
        logger.handlers = []
    fh = logging.FileHandler(LOG_FILENAME)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # Always set to DEBUG
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    # reduce scapy noise
    logging.getLogger("scapy").setLevel(logging.WARNING)
    return logger

logger = setup_logging(False)

# -------------------------
# Data classes
# -------------------------
@dataclass
class SuricataAlert:
    src_ip: str
    dest_ip: str
    signature: str
    severity: int
    timestamp: float
    signature_id: int

# -------------------------
# Suricata Manager
# -------------------------
class SuricataManager:
    """
    Start Suricata in daemon (live) mode and monitor eve.json for alerts.
    Provides:
      - set_alert_callback(callback(alert_dict))
      - start() / stop()
    """
    def __init__(self, interface: str, rules_dir: str = DEFAULT_RULES_DIR, eve_path: str = EVE_JSON_PATH):
        self.interface = interface
        self.rules_dir = rules_dir
        self.eve_path = eve_path
        self.process = None
        self.running = False
        self.alert_callback = None
        self._monitor_thread = None

    def set_alert_callback(self, cb: Callable[[dict], None]) -> None:
        self.alert_callback = cb

    def _ensure_logdir(self):
        logdir = os.path.dirname(self.eve_path) if os.path.dirname(self.eve_path) else "/var/log/suricata"
        os.makedirs(logdir, exist_ok=True)

    def start(self) -> bool:
        """Start Suricata in daemon mode. Returns True on success."""
        try:
            # Check suricata binary
            if subprocess.run(["which", "suricata"], capture_output=True).returncode != 0:
                logger.error("Suricata binary not found. Please install suricata.")
                return False

            # Ensure rules dir exists (warn if not)
            if not os.path.isdir(self.rules_dir):
                logger.warning(f"Rules directory {self.rules_dir} not found; Suricata will still try with system config.")

            self._ensure_logdir()

            # Build command: use system config and specify interface + log dir
            cmd = [
                "sudo", "suricata",
                "-c", "/etc/suricata/suricata.yaml",
                "-i", self.interface,
                "-l", os.path.dirname(self.eve_path),
                "-D"  # daemonize
            ]
            logger.info("Launching Suricata: " + " ".join(cmd))
            # start suricata as subprocess (sudo will ask for password if needed)
            # use subprocess.run to let system manage daemon or Popen then detach
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                logger.error(f"Suricata failed to launch: {proc.stderr.strip()}")
                return False

            # Start monitoring thread for eve.json
            self.running = True
            self._monitor_thread = threading.Thread(target=self._monitor_eve, daemon=True)
            self._monitor_thread.start()
            logger.info("Suricata started and EVE monitor thread running.")
            return True

        except Exception as e:
            logger.exception(f"Error starting Suricata: {e}")
            return False

    def _monitor_eve(self):
        
        TEST_MODE = True  # 🔁 Set to False once dashboard reflects real alerts properly
        logger.info(f"Monitoring EVE JSON at {self.eve_path}")
        
        # Wait for eve.json to exist
        attempts = 0
        while self.running and not os.path.exists(self.eve_path) and attempts < 60:
            attempts += 1
            time.sleep(0.5)

        if not os.path.exists(self.eve_path):
            logger.warning("EVE JSON file not found; continuing without Suricata alerts.")
            return

        try:
            with open(self.eve_path, "r") as fh:
                # Go to end initially
                fh.seek(0, os.SEEK_END)
                while self.running:
                    line = fh.readline()
                    if not line:
                        time.sleep(0.2)
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("event_type", "")
                    flow_data = data.get("flow", {})

                    # 🧠 Case 1: Normal Suricata alerts
                    if "alert" in data:
                        if self.alert_callback:
                            self.alert_callback(data)
                            logger.debug("Forwarded real Suricata alert to callback.")
                    
                    # 🧠 Case 2: Promote flow entries to synthetic alerts
                    elif event_type == "flow":
                        # Either Suricata says alerted=True, or TEST_MODE is on
                        if flow_data.get("alerted", False) or TEST_MODE:
                            synthetic_alert = {
                                "src_ip": data.get("src_ip"),
                                "dest_ip": data.get("dest_ip"),
                                "alert": {
                                    "signature": f"Flow Event {data.get('src_ip')}->{data.get('dest_ip')}",
                                    "severity": 5 if not flow_data.get("alerted", False) else 3,
                                    "signature_id": 999999
                                }
                            }
                            if self.alert_callback:
                                self.alert_callback(synthetic_alert)
                                logger.debug(f"Synthetic flow alert generated: {synthetic_alert['alert']['signature']}")
        except Exception as e:
            logger.exception(f"EVE monitor error: {e}")


    def stop(self):
        """Stop Suricata process (attempt graceful stop)"""
        if not self.running:
            return
        logger.info("Stopping Suricata...")
        try:
            # Try sending SIGTERM to suricata processes via pkill
            subprocess.run(["sudo", "pkill", "-f", "suricata"], check=False)
            time.sleep(1)
        except Exception:
            logger.exception("Error while stopping Suricata (pkill).")
        finally:
            self.running = False

# -------------------------
# -------------------------
# Network IDS
# -------------------------
class NetworkIDS:
    """Network Intrusion Detection System using Suricata for traffic analysis."""
    
    def __init__(self, interface: str = DEFAULT_INTERFACE, 
                 rules_dir: str = DEFAULT_RULES_DIR):
        """Initialize the Network IDS with Suricata integration.
        
        Args:
            interface: Network interface to monitor (e.g., 'eth0', 'lo')
            rules_dir: Directory containing Suricata rules
        """
        self.interface = interface
        self.rules_dir = rules_dir
        self.suricata = None
        
        # Deque for recent SuricataAlert objects
        self.recent_suricata_alerts: Deque[SuricataAlert] = deque()
        self.alert_lock = threading.Lock()
        self.suricata_alert_window = SURICATA_ALERT_WINDOW

        # Statistics
        self.packet_count = 0
        self.alert_count = 0
        self.start_time = time.time()

        # Initialize Suricata
        logger.info(f"Initializing Suricata on interface {interface}")
        logger.info(f"Using rules from: {rules_dir}")
        
        try:
            self.suricata = SuricataManager(
                interface=self.interface, 
                rules_dir=self.rules_dir
            )
            self.suricata.set_alert_callback(self._suricata_callback)
            
            if not self.suricata.start():
                logger.error("Failed to start Suricata. Exiting...")
                sys.exit(1)
                
            logger.info("Suricata started successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Suricata: {e}")
            sys.exit(1)

    # Suricata callback expects raw eve JSON dict
    def _suricata_callback(self, eve_obj: dict):
        try:
            # log raw eve object for debugging (comment out later)
            logger.debug(f"EVE raw event: {json.dumps(eve_obj)}")

            src_ip = eve_obj.get("src_ip") or eve_obj.get("src")
            dest_ip = eve_obj.get("dest_ip") or eve_obj.get("dest")
            signature = eve_obj.get("alert", {}).get("signature") or eve_obj.get("sig")
            severity = int(eve_obj.get("alert", {}).get("severity") or eve_obj.get("severity", 3))
            signature_id = int(eve_obj.get("alert", {}).get("signature_id") or eve_obj.get("sid", 0))
            alert = SuricataAlert(
                src_ip=str(src_ip),
                dest_ip=str(dest_ip),
                signature=str(signature),
                severity=severity,
                timestamp=time.time(),
                signature_id=signature_id
            )
            with self.alert_lock:
                self.recent_suricata_alerts.append(alert)
                self._prune_alerts()
            logger.debug(f"Suricata alert recorded: {alert.signature} {alert.src_ip}->{alert.dest_ip}")
        except Exception:
            logger.exception("Error in suricata callback")


    def _prune_alerts(self):
        now = time.time()
        while self.recent_suricata_alerts and (now - self.recent_suricata_alerts[0].timestamp > self.suricata_alert_window):
            self.recent_suricata_alerts.popleft()

    def _find_matching_suricata(self, src_ip: str, dst_ip: str) -> Optional[SuricataAlert]:
        """Return a recent SuricataAlert matching src/dst in either direction or by single IP."""
        with self.alert_lock:
            self._prune_alerts()
            # Most strict: exact match (src->dst)
            for alert in reversed(self.recent_suricata_alerts):
                if alert.src_ip == src_ip and alert.dest_ip == dst_ip:
                    logger.debug("Matched suricata alert (exact src->dst)")
                    return alert
            # Try reversed direction (dst->src)
            for alert in reversed(self.recent_suricata_alerts):
                if alert.src_ip == dst_ip and alert.dest_ip == src_ip:
                    logger.debug("Matched suricata alert (reversed dst->src)")
                    return alert
            # Fallback: match either src or dst alone (less precise)
            for alert in reversed(self.recent_suricata_alerts):
                if alert.src_ip == src_ip or alert.dest_ip == dst_ip or alert.src_ip == dst_ip or alert.dest_ip == src_ip:
                    logger.debug("Matched suricata alert (loose match by single IP)")
                    return alert
        return None


    def _is_whitelisted_ip(self, ip_str: str) -> bool:
        try:
            ip = ipaddress.ip_address(ip_str)
        except Exception:
            return False
        for net in WHITELISTED_IP_RANGES:
            if ip in ipaddress.ip_network(net):
                return True
        return False

    def _send_to_dashboard(self, packet_info):
        """Send both alerts and normal packets to the Flask dashboard via HTTP POST."""
        def send_request():
            try:
                # Always send to /update, dashboard auto-detects alerts
                response = requests.post(DASHBOARD_URL, json=packet_info, timeout=5)
                if response.status_code != 200:
                    logger.warning(f"Dashboard returned {response.status_code}: {response.text}")
                else:
                    logger.debug(f"Packet (alert={packet_info.get('is_alert')}) sent to dashboard successfully.")
            except Exception as e:
                logger.error(f"Error sending packet to dashboard: {e}")

        threading.Thread(target=send_request, daemon=True).start()
        


    def process_packet(self, pkt: scapy.packet.Packet):
        """
        Process packet through Suricata for detection and forward to dashboard
        """
        logger.debug(f"Packet received: {pkt.summary()}")
        self.packet_count += 1

        try:
            # Extract basic packet information
            if pkt.haslayer("IP"):
                ip_layer = pkt["IP"]
                src = ip_layer.src
                dst = ip_layer.dst
                proto = ip_layer.proto
                proto_name = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}.get(proto, f'Proto-{proto}')
            elif pkt.haslayer("IPv6"):
                ip_layer = pkt["IPv6"]
                src = ip_layer.src
                dst = ip_layer.dst
                proto = ip_layer.nh if hasattr(ip_layer, "nh") else 0
                proto_name = {6: 'TCPv6', 17: 'UDPv6', 58: 'ICMPv6'}.get(proto, f'Proto-{proto}')
            else:
                # Not an IP packet, skip
                return

            # Extract ports if available
            sport = 0
            dport = 0
            if pkt.haslayer("TCP"):
                sport = int(pkt["TCP"].sport)
                dport = int(pkt["TCP"].dport)
            elif pkt.haslayer("UDP"):
                sport = int(pkt["UDP"].sport)
                dport = int(pkt["UDP"].dport)

            # Check for Suricata alerts
            is_alert = False
            alert_info = {}
            
            suricata_alert = self._find_matching_suricata(src, dst)
            if suricata_alert:
                is_alert = True
                alert_info = {
                    'is_alert': True,
                    'signature': suricata_alert.signature,
                    'severity': suricata_alert.severity,
                    'category': 'suricata_alert'
                }
                self.alert_count += 1
            
            # Prepare packet info for dashboard
            packet_info = {
                'src_ip': src or '0.0.0.0',
                'src_port': sport or 0,
                'dst_ip': dst or '0.0.0.0',  # Changed from dest_ip to dst_ip
                'dst_port': dport or 0,      # Changed from dest_port to dst_port
                'protocol': proto_name,
                'length': len(pkt),
                'timestamp': datetime.now().isoformat(),
                'is_alert': is_alert,
            }
            
            # Add alert details if this is an alert
            if is_alert and alert_info:
                packet_info.update({
                    'signature': alert_info.get('signature', 'Unknown alert'),
                    'severity': alert_info.get('severity', 3),  # Default to medium severity
                    'category': alert_info.get('category', 'suricata_alert')
                })
                
                # Log alerts at info level
                log_msg = (f"🚨 SURICATA ALERT: {proto_name} {src}:{sport} -> {dst}:{dport} "
                         f"- {packet_info.get('signature', '')}")
                logger.info(log_msg)
            else:
                # Log normal traffic at debug level
                logger.debug(f"{proto_name} {src}:{sport} -> {dst}:{dport} len={len(pkt)}")
            
            # Send to dashboard
            self._send_to_dashboard(packet_info)
            
            # Skip further processing for whitelisted traffic
            if (dport in WHITELISTED_PORTS or 
                self._is_whitelisted_ip(src) or 
                self._is_whitelisted_ip(dst)):
                logger.debug(f"Whitelisted traffic: {src}->{dst} ports {sport}->{dport}")
                return

        except Exception as e:
            logger.error(f"Error processing packet: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Packet that caused error: {pkt.summary()}")

    def stop(self):
        if self.suricata:
            self.suricata.stop()

# -------------------------
# Sniffing helpers
# -------------------------
def packet_handler(pkt):
    try:
        if isinstance(ids_global, NetworkIDS):
            ids_global.process_packet(pkt)
    except Exception:
        logger.exception("packet_handler exception")

def start_sniffing(interface: str = DEFAULT_INTERFACE, count: int = 0):
    logger.info("=" * 80)
    logger.info("                 Improved Network Intrusion Detection System (Fusion)")
    logger.info("                       Starting Packet Capture")
    logger.info("=" * 80)
    logger.info(f"[+] Interface: {interface}")
    logger.info(f"[+] BPF Filter: ip or ip6")
    logger.info(f"[+] Log file: {LOG_FILENAME}")
    logger.info("[!] Press Ctrl+C to stop\n")
    scapy.sniff(iface=interface, filter="ip or ip6", prn=packet_handler, store=0, count=count)

# -------------------------
# Main
# -------------------------
ids_global = None

def main():
    parser = argparse.ArgumentParser(description="Network IDS with Suricata")
    parser.add_argument("-i", "--interface", default=DEFAULT_INTERFACE,
                      help=f"Network interface to monitor (default: {DEFAULT_INTERFACE})")
    parser.add_argument("--rules-dir", default=DEFAULT_RULES_DIR,
                      help=f"Directory containing Suricata rules (default: {DEFAULT_RULES_DIR})")
    parser.add_argument("-c", "--count", type=int, default=0,
                      help="Number of packets to capture (0 for unlimited)")
    parser.add_argument("-d", "--debug", action="store_true", default=False,
                      help="Enable debug logging")
    args = parser.parse_args()

    global logger
    logger = setup_logging(debug=args.debug)

    # Initialize IDS
    global ids_global
    try:
        logger.info(f"Starting Suricata IDS on interface {args.interface}")
        logger.info(f"Using rules from: {args.rules_dir}")
        
        ids_global = NetworkIDS(
            interface=args.interface,
            rules_dir=args.rules_dir
        )
        
        logger.info("Press Ctrl+C to stop...")
        start_sniffing(interface=args.interface, count=args.count)
        
    except KeyboardInterrupt:
        logger.info("\nStopping Suricata IDS...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Detailed error:")
        return 1
    finally:
        if 'ids_global' in globals() and ids_global:
            ids_global.stop()
    
    # Summary
    if 'ids_global' in globals() and ids_global:
        elapsed = time.time() - ids_global.start_time
        logger.info("=" * 80)
        logger.info("Packet Capture Summary:")
        logger.info("=" * 80)
        logger.info(f"Total packets processed: {ids_global.packet_count}")
        logger.info(f"Total alerts detected: {ids_global.alert_count}")
        logger.info(f"Duration: {elapsed:.2f} seconds")
        if elapsed > 0:
            logger.info(f"Average packets/second: {ids_global.packet_count/elapsed:.2f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
