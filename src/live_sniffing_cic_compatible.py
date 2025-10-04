"""
Live Network Sniffer Compatible with CIC-Trained Models
This version can use models trained on CIC datasets for real-time detection
"""

import os
import csv
import argparse
import logging
import joblib
import pandas as pd
from scapy.all import sniff
from collections import Counter, defaultdict
from feature_extractor_patched_final import extract_features, flow_tracker

# Configuration
DEFAULT_INTERFACE = "eth0"
LOG_FILENAME = "sniffing_log_cic.txt"
DEBUG_LOG_FILE = "debug_log_cic.csv"
CSV_FILENAME = "captured_features_cic.csv"

# Global variables
DEBUG_HEADER_WRITTEN = False
CSV_HEADER_WRITTEN = False
initialized = False
label_counter = Counter()
model = None
label_encoder = None
scaler = None
selected_features = None


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
    global model, label_encoder, scaler, selected_features
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    if encoder_path and os.path.exists(encoder_path):
        print(f"Loading label encoder from {encoder_path}...")
        label_encoder = joblib.load(encoder_path)
    
    if scaler_path and os.path.exists(scaler_path):
        print(f"Loading scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)
    
    if features_path and os.path.exists(features_path):
        print(f"Loading selected features from {features_path}...")
        with open(features_path, 'r') as f:
            selected_features = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(selected_features)} features")
    
    print("✓ Model loaded successfully")


def map_pcap_features_to_cic(pcap_features: dict) -> dict:
    """
    Map PCAP-extracted features to CIC-like features
    This is a best-effort mapping - not all CIC features can be extracted from live packets
    
    Args:
        pcap_features: Features extracted from PCAP
    
    Returns:
        Dictionary with CIC-compatible features
    """
    cic_features = {}
    
    # Direct mappings
    cic_features['flow_duration'] = pcap_features.get('flow_duration', 0)
    cic_features['total_fwd_packets'] = pcap_features.get('flow_packet_count', 0)
    cic_features['total_backward_packets'] = 0  # Not available from single packet
    cic_features['total_length_of_fwd_packets'] = pcap_features.get('src_bytes', 0)
    cic_features['total_length_of_bwd_packets'] = pcap_features.get('dst_bytes', 0)
    
    # Packet length features
    cic_features['fwd_packet_length_max'] = pcap_features.get('src_bytes', 0)
    cic_features['fwd_packet_length_min'] = pcap_features.get('src_bytes', 0)
    cic_features['fwd_packet_length_mean'] = pcap_features.get('src_bytes', 0)
    cic_features['fwd_packet_length_std'] = 0
    
    # Flow features
    cic_features['flow_bytes/s'] = pcap_features.get('packet_rate', 0) * pcap_features.get('src_bytes', 0)
    cic_features['flow_packets/s'] = pcap_features.get('packet_rate', 0)
    
    # TCP flags
    cic_features['fin_flag_count'] = pcap_features.get('tcp_flag_FIN', 0)
    cic_features['syn_flag_count'] = pcap_features.get('tcp_flag_SYN', 0)
    cic_features['rst_flag_count'] = pcap_features.get('tcp_flag_RST', 0)
    cic_features['psh_flag_count'] = pcap_features.get('tcp_flag_PSH', 0)
    cic_features['ack_flag_count'] = pcap_features.get('tcp_flag_ACK', 0)
    cic_features['urg_flag_count'] = pcap_features.get('tcp_flag_URG', 0)
    cic_features['ece_flag_count'] = pcap_features.get('tcp_flag_ECE', 0)
    cic_features['cwe_flag_count'] = pcap_features.get('tcp_flag_CWR', 0)
    
    # Protocol
    cic_features['protocol'] = 6 if pcap_features.get('protocol_type_tcp', 0) == 1 else \
                               17 if pcap_features.get('protocol_type_udp', 0) == 1 else \
                               1 if pcap_features.get('protocol_type_icmp', 0) == 1 else 0
    
    # Ports
    cic_features['source_port'] = pcap_features.get('tcp_sport', 0) or pcap_features.get('udp_sport', 0)
    cic_features['destination_port'] = pcap_features.get('tcp_dport', 0) or pcap_features.get('udp_dport', 0)
    
    # Additional features (approximations)
    cic_features['average_packet_size'] = pcap_features.get('src_bytes', 0)
    cic_features['fwd_header_length'] = 20  # Typical IP header
    cic_features['bwd_header_length'] = 0
    
    # Entropy
    cic_features['payload_entropy'] = pcap_features.get('payload_entropy', 0)
    
    # Fill remaining features with zeros (features we can't extract from live traffic)
    default_features = [
        'bwd_packet_length_max', 'bwd_packet_length_min', 'bwd_packet_length_mean',
        'bwd_packet_length_std', 'min_packet_length', 'max_packet_length',
        'packet_length_mean', 'packet_length_std', 'packet_length_variance',
        'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',
        'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min',
        'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',
        'down/up_ratio', 'init_win_bytes_forward', 'init_win_bytes_backward',
        'act_data_pkt_fwd', 'min_seg_size_forward', 'active_mean', 'active_std',
        'active_max', 'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min'
    ]
    
    for feature in default_features:
        if feature not in cic_features:
            cic_features[feature] = 0
    
    return cic_features


def predict_with_cic_model(pcap_features: dict) -> str:
    """
    Predict using CIC-trained model
    
    Args:
        pcap_features: Features extracted from PCAP
    
    Returns:
        Predicted label
    """
    try:
        # Map PCAP features to CIC format
        cic_features = map_pcap_features_to_cic(pcap_features)
        
        # Create DataFrame
        df = pd.DataFrame([cic_features])
        
        # Select only the features used during training
        if selected_features:
            # Add missing features as zeros
            for feat in selected_features:
                if feat not in df.columns:
                    df[feat] = 0
            df = df[selected_features]
        
        # Scale features if scaler is available
        if scaler:
            df = pd.DataFrame(
                scaler.transform(df),
                columns=df.columns
            )
        
        # Predict
        prediction = model.predict(df)[0]
        
        # Decode label if encoder is available
        if label_encoder:
            label = label_encoder.inverse_transform([prediction])[0]
        else:
            label = str(prediction)
        
        return label
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return "unknown"


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


def packet_handler(pkt, use_cic_model=True, pcap_model=None, pcap_encoder=None):
    """
    Handle each captured packet
    
    Args:
        pkt: Scapy packet
        use_cic_model: Whether to use CIC-trained model
        pcap_model: PCAP-trained model (if not using CIC)
        pcap_encoder: PCAP label encoder
    """
    global initialized, CSV_HEADER_WRITTEN
    
    try:
        if not pkt.haslayer("IP"):
            return
        
        ip = pkt["IP"]
        proto = "tcp" if pkt.haslayer("TCP") else "udp" if pkt.haslayer("UDP") else "icmp"
        sport = pkt.sport if hasattr(pkt, "sport") else 0
        dport = pkt.dport if hasattr(pkt, "dport") else 0
        flow_id = f"{ip.src}-{ip.dst}-{proto}-{sport}-{dport}"
        
        # Extract features using existing feature extractor
        features = extract_features(pkt, flow_tracker, flow_id)
        
        # Predict using appropriate model
        if use_cic_model and model is not None:
            prediction = predict_with_cic_model(features)
        elif pcap_model is not None:
            prediction = predict_with_pcap_model(features, pcap_model, pcap_encoder)
        else:
            prediction = "no_model_loaded"
        
        features['prediction'] = prediction
        
        # Initialize CSV if needed
        if not initialized:
            with open(CSV_FILENAME, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=features.keys())
                writer.writeheader()
            initialized = True
        
        # Save features
        with open(CSV_FILENAME, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=features.keys())
            writer.writerow(features)
        
        # Log
        logging.info(f"Packet: {ip.src}:{sport} -> {ip.dst}:{dport} | Prediction: {prediction}")
        label_counter[prediction] += 1
        
        if len(label_counter) > 0 and sum(label_counter.values()) % 10 == 0:
            logging.info(f"Summary: {dict(label_counter)}")
        
        # Debug log
        log_debug_info(features, prediction)
        
    except Exception as e:
        logging.warning(f"Error in packet_handler: {e}")


def start_sniffing(interface=DEFAULT_INTERFACE, packet_count=0, bpf_filter="ip",
                  use_cic_model=True, pcap_model=None, pcap_encoder=None):
    """Start packet sniffing"""
    logging.info(f"Sniffing started on interface: {interface}, Filter: '{bpf_filter}'")
    logging.info(f"Using {'CIC-trained' if use_cic_model else 'PCAP-trained'} model")
    
    try:
        sniff(
            iface=interface,
            prn=lambda pkt: packet_handler(pkt, use_cic_model, pcap_model, pcap_encoder),
            store=False,
            filter=bpf_filter,
            count=packet_count
        )
    except Exception as e:
        logging.critical(f"Sniffing failed: {e}")
    
    logging.info("Sniffing session ended.")
    logging.info(f"Final summary: {dict(label_counter)}")


def main():
    parser = argparse.ArgumentParser(
        description="Live Network Sniffer with CIC Model Support",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Sniffing options
    parser.add_argument("-i", "--interface", type=str, default=DEFAULT_INTERFACE,
                       help="Network interface (default: eth0)")
    parser.add_argument("-c", "--count", type=int, default=0,
                       help="Number of packets to sniff (0 = infinite)")
    parser.add_argument("-f", "--filter", type=str, default="ip",
                       help="BPF filter string (default: ip)")
    
    # Model options
    parser.add_argument("--model-type", type=str, choices=["cic", "pcap"], default="cic",
                       help="Type of model to use")
    parser.add_argument("--model-path", type=str,
                       default="models/cic_models/random_forest_model.pkl",
                       help="Path to model file")
    parser.add_argument("--encoder-path", type=str,
                       default="artifacts/cic_label_encoder.pkl",
                       help="Path to label encoder")
    parser.add_argument("--scaler-path", type=str,
                       default="artifacts/cic_scaler.pkl",
                       help="Path to feature scaler")
    parser.add_argument("--features-path", type=str,
                       default="artifacts/selected_features.txt",
                       help="Path to selected features list")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        filename=LOG_FILENAME,
        filemode='a',
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', "%H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    # Load model
    if args.model_type == "cic":
        load_cic_model(
            model_path=args.model_path,
            encoder_path=args.encoder_path,
            scaler_path=args.scaler_path,
            features_path=args.features_path
        )
        start_sniffing(
            interface=args.interface,
            packet_count=args.count,
            bpf_filter=args.filter,
            use_cic_model=True
        )
    else:
        # Load PCAP model
        pcap_model = joblib.load("models/xgboost_model_final.pkl")
        pcap_encoder = joblib.load("models/label_encoder_xgboost.pkl")
        start_sniffing(
            interface=args.interface,
            packet_count=args.count,
            bpf_filter=args.filter,
            use_cic_model=False,
            pcap_model=pcap_model,
            pcap_encoder=pcap_encoder
        )


if __name__ == "__main__":
    main()
