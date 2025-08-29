import joblib
import pandas as pd
from scapy.all import rdpcap
from feature_extractor import extract_features, flow_tracker
from collections import defaultdict
from scapy.layers.inet import IP, TCP, UDP, ICMP

# Load model and encoder
model = joblib.load("../models/xgboost_model_final.pkl")
label_encoder = joblib.load("../models/label_encoder_xgboost.pkl")

# Label mapping
label_mapping = {
    0: "back", 1: "buffer_overflow", 2: "ftp_write", 3: "guess_passwd", 4: "imap",
    5: "ipsweep", 6: "land", 7: "loadmodule", 8: "multihop", 9: "neptune",
    10: "nmap", 11: "normal", 12: "perl", 13: "phf", 14: "pod", 15: "portsweep",
    16: "rootkit", 17: "satan", 18: "smurf", 19: "spy", 20: "teardrop",
    21: "warezclient", 22: "warezmaster"
}

expected_columns = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'protocol_type_icmp', 'protocol_type_tcp',
    'protocol_type_udp', 'service_domain_u', 'service_eco_i', 'service_ecr_i', 'service_finger',
    'service_ftp', 'service_ftp_data', 'service_http', 'service_other', 'service_private',
    'service_rare', 'service_smtp', 'service_telnet', 'flag_OTH', 'flag_REJ', 'flag_RSTO',
    'flag_RSTOS0', 'flag_RSTR', 'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH']


# Define prediction wrapper
def predict_label(features):
    df = pd.DataFrame([features])
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]
    df = df.astype(float)
    pred = model.predict(df)[0]
    return label_mapping.get(pred, str(pred))

# Load pcap
packets = rdpcap("test_traffic.pcap")
flow_tracker.clear()

print("=== Predictions on PCAP ===")
for pkt in packets:
    if not pkt.haslayer(IP):
        continue
    ip = pkt[IP]
    proto = "tcp" if pkt.haslayer(TCP) else "udp" if pkt.haslayer(UDP) else "icmp"
    sport = pkt.sport if hasattr(pkt, "sport") else 0
    dport = pkt.dport if hasattr(pkt, "dport") else 0
    flow_id = f"{ip.src}-{ip.dst}-{proto}-{sport}-{dport}"
    
    features = extract_features(pkt, flow_tracker, flow_id)
    if features:
        label = predict_label(features)
        print(f"Packet from {ip.src} to {ip.dst}: prediction = {label}")
