import os
import csv
import argparse
import logging
from scapy.all import sniff
import pandas as pd
import joblib
from collections import Counter
from feature_extractor_patched_final import extract_features, init_csv, flow_tracker, CSV_FILENAME


DEBUG_LOG_FILE = "debug_log.csv"
DEBUG_HEADER_WRITTEN = False
model = joblib.load("../models/xgboost_model_final.pkl")
label_encoder = joblib.load("../models/label_encoder_xgboost.pkl")
label_counter = Counter()


DEFAULT_INTERFACE = "eth0"
LOG_FILENAME = "sniffing_log.txt"

label_mapping = {
    0: "back",
    1: "buffer_overflow",
    2: "ftp_write",
    3: "guess_passwd",
    4: "imap",
    5: "ipsweep",
    6: "land",
    7: "loadmodule",
    8: "multihop",
    9: "neptune",
    10: "nmap",
    11: "normal",
    12: "perl",
    13: "phf",
    14: "pod",
    15: "portsweep",
    16: "rootkit",
    17: "satan",
    18: "smurf",
    19: "spy",
    20: "teardrop",
    21: "warezclient",
    22: "warezmaster"
}

def log_debug_info(features, prediction_label):
    global DEBUG_HEADER_WRITTEN

    features_with_prediction = dict(features)
    features_with_prediction["prediction"] = prediction_label

    with open(DEBUG_LOG_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=features_with_prediction.keys())
        if not DEBUG_HEADER_WRITTEN:
            writer.writeheader()
            DEBUG_HEADER_WRITTEN = True
        writer.writerow(features_with_prediction)

def predict_label(feature_dict):
    try:
        df = pd.DataFrame([feature_dict])
        if 'label' in df.columns:
            df = df.drop(columns=['label'])

        # Patch: Ensure all expected columns exist and are ordered
        expected_columns = ["duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
            "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
            "num_access_files", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
            "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate", "protocol_type_icmp", "protocol_type_tcp", "protocol_type_udp", "service_domain_u",
            "service_eco_i", "service_ecr_i", "service_finger", "service_ftp", "service_ftp_data", "service_http",
            "service_other", "service_private", "service_rare", "service_smtp", "service_telnet", "flag_OTH", "flag_REJ",
            "flag_RSTO", "flag_RSTOS0", "flag_RSTR", "flag_S0", "flag_S1", "flag_S2", "flag_S3", "flag_SF", "flag_SH"]

        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]

        prediction = model.predict(df)[0]
        label_name = label_mapping.get(prediction, str(prediction))  
        return label_name
    except Exception as e:
        print(f"[!] Prediction error: {e}")
        return "unknown"


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


initialized = False


def packet_handler(pkt):
    global initialized

    try:
       
        if not pkt.haslayer("IP"):
            return
        ip = pkt["IP"]
        proto = "tcp" if pkt.haslayer("TCP") else "udp" if pkt.haslayer("UDP") else "icmp"
        sport = pkt.sport if hasattr(pkt, "sport") else 0
        dport = pkt.dport if hasattr(pkt, "dport") else 0
        flow_id = f"{ip.src}-{ip.dst}-{proto}-{sport}-{dport}"

        
        features = extract_features(
            pkt,
            flow_tracker,
            flow_id,
            predict_label_fn=predict_label,
            save_features=True
        )

        if not initialized:
            init_csv(CSV_FILENAME, features.keys())
            initialized = True

        logging.info(f"Packet from {ip.src} to {ip.dst}, prediction = {features.get('prediction', 'N/A')}")
        label = features.get('prediction', 'unknown')
        label_counter[label] += 1
        logging.info(f"Summary count so far: {dict(label_counter)}")
        log_debug_info(features, label)


    except Exception as e:
        logging.warning(f"Error in packet_handler: {e}")


def start_sniffing(interface=DEFAULT_INTERFACE, packet_count=0, bpf_filter="ip"):
    logging.info(f"Sniffing started on interface: {interface}, Filter: '{bpf_filter}'")
    try:
        sniff(
            iface=interface,
            prn=packet_handler,
            store=False,
            filter=bpf_filter,
            count=packet_count
        )
    except Exception as e:
        logging.critical(f"Sniffing failed: {e}")
    logging.info("Sniffing session ended.")


def get_args():
    parser = argparse.ArgumentParser(description="Live Network Packet Sniffer")
    parser.add_argument("-i", "--interface", type=str, default=DEFAULT_INTERFACE, help="Network interface (default: eth0)")
    parser.add_argument("-c", "--count", type=int, default=0, help="Number of packets to sniff (0 = infinite)")
    parser.add_argument("-f", "--filter", type=str, default="ip", help="BPF filter string (default: ip)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    start_sniffing(interface=args.interface, packet_count=args.count, bpf_filter=args.filter)




