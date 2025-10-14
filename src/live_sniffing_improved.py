
#!/usr/bin/env python3
"""
live_sniffing_improved_fusion.py
Integrated version of live_sniffing_improved.py with:
 - Proper SuricataManager running Suricata in daemon mode (-D)
 - EVE JSON monitoring thread
 - Hybrid ML + Suricata fusion logic (Option 4)
 - Keeps existing model loading/prediction pipeline
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import logging
import argparse
import warnings
import threading
import ipaddress
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable, Set, Union
from pathlib import Path
import subprocess
import json
from collections import deque
from dataclasses import dataclass

import scapy.all as scapy

# -------------------------
# Configuration & Constants
# -------------------------
DEFAULT_INTERFACE = "eth0"
LOG_FILENAME = "ids_improved.log"
DEFAULT_MODEL_PATH = str(Path(__file__).parent.parent / "models" / "xgboost_model_cic.pkl")
DEFAULT_RULES_DIR = "/var/lib/suricata/rules"
EVE_JSON_PATH = "/var/log/suricata/eve.json"

# Fusion parameters
SURICATA_ALERT_WINDOW = 30       # seconds to consider a recent suricata alert relevant
ML_CONFIDENCE_HIGH = 0.85       # ML high confidence threshold
ML_CONFIDENCE_MED = 0.7         # ML medium confidence threshold

# Whitelists and false-positive suppression (minimal defaults)
WHITELISTED_PORTS = {80, 443, 53, 123}
WHITELISTED_IP_RANGES = {
    "172.16.0.0/12",
    "192.168.0.0/16",
    "10.0.0.0/8",
}

# -------------------------
# Utilities
# -------------------------
def setup_logging(debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("nids_improved_fusion")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # Clear existing handlers
    if logger.handlers:
        logger.handlers = []
    fh = logging.FileHandler(LOG_FILENAME)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
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
        """Tail the EVE JSON file and forward alert objects to callback"""
        logger.info(f"Monitoring EVE JSON at {self.eve_path}")
        # wait for file to exist
        attempts = 0
        while self.running and not os.path.exists(self.eve_path) and attempts < 60:
            attempts += 1
            time.sleep(0.5)

        if not os.path.exists(self.eve_path):
            logger.warning("EVE JSON file not found; continuing without Suricata alerts.")
            return

        try:
            with open(self.eve_path, "r") as fh:
                # go to end initially
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
                    # Only forward alerts (eve can contain many event types)
                    if "alert" in data:
                        if self.alert_callback:
                            try:
                                self.alert_callback(data)
                            except Exception:
                                logger.exception("Alert callback raised an exception")
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
# Model loading / prediction helpers
# -------------------------
model = None
scaler = None
selected_features = []

def load_model(model_path: str, scaler_path: Optional[str] = None) -> bool:
    global model, scaler, selected_features
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    try:
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict):
            model = model_data.get("model", None)
            selected_features = model_data.get("feature_names", [])
        else:
            model = model_data
            selected_features = []
        if scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        logger.info(f"Model loaded: {getattr(model, '__class__', 'unknown')}")
        return True
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        return False

def preprocess_features(packet_features: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Prepare feature DataFrame for model prediction"""
    if packet_features is None:
        return None
    try:
        df = pd.DataFrame([packet_features])
        expected = getattr(model, "feature_names_in_", None)
        if expected is None and selected_features:
            expected = selected_features
        if expected is not None and len(expected) > 0:
            missing = set(expected) - set(df.columns)
            for m in missing:
                df[m] = 0
            df = df[list(expected)]
        if scaler is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X = scaler.transform(df)
                df = pd.DataFrame(X, columns=df.columns)
        return df
    except Exception:
        logger.exception("Error preprocessing features")
        return None

def predict_label_and_confidence(packet_features: Dict[str, Any]) -> Tuple[str, float]:
    """Return label (string) and confidence (0-1)"""
    if model is None:
        return "model_not_loaded", 0.0
    df = preprocess_features(packet_features)
    if df is None or df.empty:
        return "error", 0.0
    if not hasattr(model, "predict_proba"):
        pred = model.predict(df)[0]
        return str(pred), 1.0
    probs = model.predict_proba(df)[0]
    idx = int(np.argmax(probs))
    label = str(model.classes_[idx]) if hasattr(model, "classes_") else str(idx)
    confidence = float(probs[idx])
    return label, confidence

# -------------------------
# Network IDS with fusion
# -------------------------
class NetworkIDS:
    def __init__(self, interface: str = DEFAULT_INTERFACE, rules_dir: str = DEFAULT_RULES_DIR,
                 use_suricata: bool = True):
        self.interface = interface
        self.rules_dir = rules_dir
        self.use_suricata = use_suricata
        self.suricata = None
        # deque for recent SuricataAlert objects
        self.recent_suricata_alerts = deque()
        self.alert_lock = threading.Lock()
        self.suricata_alert_window = SURICATA_ALERT_WINDOW

        # Stats
        self.packet_count = 0
        self.attack_count = 0
        self.start_time = time.time()

        # Initialize Suricata if requested
        if self.use_suricata:
            self.suricata = SuricataManager(interface=self.interface, rules_dir=self.rules_dir)
            # set callback
            self.suricata.set_alert_callback(self._suricata_callback)
            started = self.suricata.start()
            if not started:
                logger.warning("Suricata could not be started; continuing without Suricata alerts.")
                self.use_suricata = False

    # Suricata callback expects raw eve JSON dict
    def _suricata_callback(self, eve_obj: dict):
        try:
            # Map fields that may be in different locations depending on event format
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
                # prune old alerts
                self._prune_alerts()
            logger.debug(f"Suricata alert recorded: {alert.signature} {alert.src_ip}->{alert.dest_ip}")
        except Exception:
            logger.exception("Error in suricata callback")

    def _prune_alerts(self):
        now = time.time()
        while self.recent_suricata_alerts and (now - self.recent_suricata_alerts[0].timestamp > self.suricata_alert_window):
            self.recent_suricata_alerts.popleft()

    def _find_matching_suricata(self, src_ip: str, dst_ip: str) -> Optional[SuricataAlert]:
        """Return a recent SuricataAlert matching src/dst (or None)"""
        with self.alert_lock:
            self._prune_alerts()
            for alert in reversed(self.recent_suricata_alerts):
                if alert.src_ip == src_ip and alert.dest_ip == dst_ip:
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

    def process_packet(self, pkt: scapy.packet.Packet):
        """
        Core processing per packet: extract features, get ML prediction,
        check Suricata cache, then fuse decisions.
        """
        logger.debug(f"Packet received: {pkt.summary()}")
        self.packet_count += 1

        # basic extraction: best-effort
        try:
            # IPv4 or IPv6
            if pkt.haslayer("IP"):
                ip_layer = pkt["IP"]
                src = ip_layer.src
                dst = ip_layer.dst
                proto = ip_layer.proto
            elif pkt.haslayer("IPv6"):
                ip_layer = pkt["IPv6"]
                src = ip_layer.src
                dst = ip_layer.dst
                proto = ip_layer.nh if hasattr(ip_layer, "nh") else 0
            else:
                # not IP packet
                return

            sport = None
            dport = None
            if pkt.haslayer("TCP"):
                sport = int(pkt["TCP"].sport)
                dport = int(pkt["TCP"].dport)
            elif pkt.haslayer("UDP"):
                sport = int(pkt["UDP"].sport)
                dport = int(pkt["UDP"].dport)

            # quick whitelist checks
            if dport in WHITELISTED_PORTS or self._is_whitelisted_ip(src) or self._is_whitelisted_ip(dst):
                logger.debug(f"Whitelisted traffic: {src}->{dst} ports {sport}->{dport}")
                return

            # assemble feature dict - adapt to your feature extractor
            features = {
                "src_ip": src,
                "dst_ip": dst,
                "src_port": int(sport) if sport is not None else 0,
                "dst_port": int(dport) if dport is not None else 0,
                "protocol_num": int(proto) if proto is not None else 0,
                "length": len(bytes(pkt)) if hasattr(pkt, "build") or hasattr(pkt, "original") else 0,
            }

            # ML prediction
            label, confidence = predict_label_and_confidence(features)

            # Check Suricata recent alerts
            suricata_alert = None
            if self.use_suricata:
                suricata_alert = self._find_matching_suricata(src, dst)

            # Fusion logic
            if suricata_alert and confidence >= ML_CONFIDENCE_MED:
                # Confirmed attack (both signals)
                logger.warning(f"🚨 CONFIRMED ATTACK: {suricata_alert.signature} {src}->{dst} (ML_conf={confidence:.2f})")
                self.attack_count += 1
            elif suricata_alert:
                # Suricata-only (log but do not treat as fully confirmed)
                logger.info(f"⚠️ Suricata alert only: {suricata_alert.signature} {src}->{dst} (ML_conf={confidence:.2f})")
            elif confidence >= ML_CONFIDENCE_HIGH:
                # ML-only high confidence
                logger.warning(f"🤖 ML-only anomaly: label={label} conf={confidence:.2f} {src}->{dst}")
                self.attack_count += 1
            else:
                # benign or low-confidence ML
                logger.debug(f"Normal/low-confidence: label={label} conf={confidence:.2f} {src}->{dst}")

        except Exception:
            logger.exception("Error processing packet")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interface", default=DEFAULT_INTERFACE)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scaler", default=None)
    parser.add_argument("--rules-dir", default=DEFAULT_RULES_DIR)
    parser.add_argument("--no-suricata", action="store_true")
    parser.add_argument("-c", "--count", type=int, default=0)
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    global logger
    logger = setup_logging(debug=args.debug)

    logger.info("Starting Network IDS (fusion)")
    logger.info(f"Interface: {args.interface}")
    logger.info(f"Suricata: {'disabled' if args.no_suricata else 'enabled'}")
    if os.geteuid() != 0:
        logger.error("This script requires root privileges. Run with sudo.")
        return 1

    # load model
    if not load_model(args.model, args.scaler):
        logger.error("Model load failed - exiting")
        return 1

    use_suricata = not args.no_suricata
    global ids_global
    ids_global = NetworkIDS(interface=args.interface, rules_dir=args.rules_dir, use_suricata=use_suricata)

    try:
        start_sniffing(interface=args.interface, count=args.count)
    except KeyboardInterrupt:
        logger.info("Stopping due to KeyboardInterrupt")
    except Exception:
        logger.exception("Fatal error in main")
    finally:
        if ids_global:
            ids_global.stop()
        # Summary
        if ids_global:
            elapsed = time.time() - ids_global.start_time
            logger.info("=" * 80)
            logger.info("Packet Capture Summary:")
            logger.info("=" * 80)
            logger.info(f"Total packets processed: {ids_global.packet_count}")
            logger.info(f"Total attacks detected: {ids_global.attack_count}")
            logger.info(f"Duration: {elapsed:.2f} seconds")
            if elapsed > 0:
                logger.info(f"Average packets/second: {ids_global.packet_count / elapsed:.2f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
