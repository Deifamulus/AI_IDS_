
import argparse
import csv
import time
from collections import defaultdict
from scapy.all import sniff, Raw
from socket import getservbyport
import os
from scapy.layers.inet import IP, TCP, UDP, ICMP

CSV_FILENAME = "captured_features.csv"
CSV_HEADER_WRITTEN = False

SERVICE_PORTS = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet",
    25: "smtp", 53: "domain_u", 69: "tftp", 80: "http",
    110: "pop3", 143: "imap", 443: "https"
}
DEFAULT_SERVICE = "other"

def get_service(pkt):
    try:
        if pkt.haslayer(TCP):
            dport = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            dport = pkt[UDP].dport
        else:
            return DEFAULT_SERVICE
        return SERVICE_PORTS.get(dport, getservbyport(dport))
    except:
        return DEFAULT_SERVICE

def get_protocol(pkt):
    if pkt.haslayer(TCP):
        return "tcp"
    elif pkt.haslayer(UDP):
        return "udp"
    elif pkt.haslayer(ICMP):
        return "icmp"
    else:
        return "other"

def get_flag(pkt):
    if not pkt.haslayer(TCP):
        return "OTH"
    try:
        flags = pkt[TCP].flags
        if flags == 0x02: return "S0"      # SYN
        if flags == 0x12: return "S1"      # SYN-ACK
        if flags == 0x14: return "REJ"     # RST-ACK
        if flags == 0x04: return "RSTR"    # RST
        if flags == 0x01: return "SF"      # FIN
        if flags == 0x11: return "S2"      # FIN-ACK
        if flags == 0x03: return "SH"      # SYN-FIN
        return "OTH"
    except:
        return "OTH"

flow_tracker = defaultdict(lambda: {
    "packet_count": 0,
    "byte_count": 0,
    "timestamps": [],
    "flags": {"SYN": 0, "ACK": 0, "FIN": 0, "RST": 0},
})


def init_csv(file_path, sample_feature_keys):
    global CSV_HEADER_WRITTEN
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sample_feature_keys)
        if not file_exists or not CSV_HEADER_WRITTEN:
            writer.writeheader()
            CSV_HEADER_WRITTEN = True

def save_features_to_csv(feature_dict, file_path):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=feature_dict.keys())
        writer.writerow(feature_dict)

def extract_features(pkt, tracker, flow_id, predict_label_fn=None, save_features=False):
    if not IP in pkt:
        return None

    features = {}
    ip_layer = pkt[IP]
    proto = get_protocol(pkt)
    features["src_ip"] = ip_layer.src
    features["dst_ip"] = ip_layer.dst
    features["land"] = int(ip_layer.src == ip_layer.dst)

    # Add all NSL-KDD dummy (defaulted) fields
    for col in ["wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
                "num_compromised", "root_shell", "su_attempted", "num_root",
                "num_file_creations", "num_shells", "num_access_files",
                "is_host_login", "is_guest_login"]:
        features[col] = 0

    if TCP in pkt or UDP in pkt:
        transport = pkt[TCP] if TCP in pkt else pkt[UDP]
        features["sport"] = transport.sport
        features["dport"] = transport.dport
    else:
        features["sport"] = 0
        features["dport"] = 0

    features["packet_size"] = len(pkt)
    features["ttl"] = ip_layer.ttl
    features["src_bytes"] = len(pkt[Raw].load) if pkt.haslayer(Raw) else 0
    features["dst_bytes"] = 0  # No reverse tracking

    # Time-based tracking
    tracker[flow_id]["packet_count"] += 1
    tracker[flow_id]["byte_count"] += len(pkt)
    pkt_time = float(pkt.time)
    tracker[flow_id]["timestamps"].append(pkt_time)
    timestamps = tracker[flow_id]["timestamps"]
    features["duration"] = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0

    # Statistical approximations
    for field in [
        "count", "srv_count", "dst_host_count", "dst_host_srv_count"
    ]:
        features[field] = tracker[flow_id]["packet_count"]

    for field in [
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]:
        features[field] = 0.0

    features["same_srv_rate"] = 1.0
    features["dst_host_same_srv_rate"] = 1.0

    # One-hot encoding
    for p in ["tcp", "udp", "icmp"]:
        features[f"protocol_type_{p}"] = 1 if proto == p else 0

    service = get_service(pkt)
    for s in ["domain_u", "eco_i", "ecr_i", "finger", "ftp", "ftp_data", "http", "other",
              "private", "rare", "smtp", "telnet"]:
        features[f"service_{s}"] = 1 if service == s else 0

    flag = get_flag(pkt)
    for f in ["OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "S0", "S1", "S2", "S3", "SF", "SH"]:
        features[f"flag_{f}"] = 1 if flag == f else 0

    if predict_label_fn:
        features["prediction"] = predict_label_fn(features)

    if save_features:
        save_features_to_csv(features, CSV_FILENAME)

    return features

def packet_callback(pkt, features_list, label):
    if not pkt.haslayer(IP):
        return
    try:
        ip = pkt[IP]
        proto = get_protocol(pkt)
        sport = pkt.sport if hasattr(pkt, "sport") else 0
        dport = pkt.dport if hasattr(pkt, "dport") else 0
        flow_id = f"{ip.src}-{ip.dst}-{proto}-{sport}-{dport}"
        features = extract_features(pkt, flow_tracker, flow_id)
        if features:
            features["label"] = label
            features_list.append(features)
    except Exception as e:
        print(f"[!] Error processing packet: {e}")

def extract_features_from_pcap(pcap_file, label="normal", output_dir="extracted_csv"):
    from scapy.all import rdpcap

    packets = rdpcap(pcap_file)
    features_list = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for pkt in packets:
        if not pkt.haslayer(IP):
            continue
        try:
            proto = get_protocol(pkt)
            sport = pkt.sport if hasattr(pkt, "sport") else 0
            dport = pkt.dport if hasattr(pkt, "dport") else 0
            flow_id = f"{pkt[IP].src}-{pkt[IP].dst}-{proto}-{sport}-{dport}"
            features = extract_features(pkt, flow_tracker, flow_id)
            if features:
                features["label"] = label
                features_list.append(features)
        except Exception as e:
            print(f"[!] Error processing packet: {e}")

    if features_list:
        out_csv = os.path.join(output_dir, f"{label}.csv")
        keys = features_list[0].keys()
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(features_list)
        print(f"[✓] Extracted {len(features_list)} packets from {pcap_file} into {out_csv}")
        return out_csv
    else:
        print(f"[!] No usable packets found in {pcap_file}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Packet Feature Extractor")
    parser.add_argument("input_pcap", type=str, help="Path to the input PCAP file")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=str, default="features.csv")
    parser.add_argument("--label", type=str, choices=["normal", "attack"], default="normal")
    args = parser.parse_args()

    features_list = []
    sniff(count=args.count, prn=lambda pkt: packet_callback(pkt, features_list, args.label), store=False)

    if features_list:
        keys = features_list[0].keys()
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(features_list)
        print(f"[✓] Saved {len(features_list)} packets to {args.output}")
    else:
        print("[!] No features captured.")

if __name__ == "__main__":
    main()