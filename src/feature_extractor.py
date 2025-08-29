import argparse
import csv
import math
import time
from collections import defaultdict
from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
from socket import getservbyport
import os

CSV_FILENAME = "captured_features.csv"
CSV_HEADER_WRITTEN = False

SERVICE_PORTS = {
    20: "ftp",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "domain",
    69: "tftp",
    80: "http",
    110: "pop3",
    143: "imap",
    443: "https",
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

        return SERVICE_PORTS.get(dport, get_service_by_port(dport))
    except:
        return DEFAULT_SERVICE


def get_service_by_port(port):
    try:
        return getservbyport(port)
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
    if pkt.haslayer(TCP):
        flags = pkt[TCP].flags
        if flags == "S":
            return "S0"
        elif flags == "SA":
            return "S1"
        elif flags == "RA":
            return "REJ"
        elif flags == "R":
            return "RSTR"
        elif flags == "F":
            return "SF"
        elif flags == "FA":
            return "S2"
        elif flags == "SH":
            return "SH"
        else:
            return "OTH"
    return "OTH"


flow_tracker = defaultdict(lambda: {"timestamps": [], "packet_count": 0})


def calculate_entropy(data):
    if not data:
        return 0.0
    entropy = 0
    length = len(data)
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


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


def extract_features(pkt, flow_tracker, flow_id, predict_label_fn=None, save_features=False):
    features = {}

    
    pkt_time = pkt.time if hasattr(pkt, "time") else time.time()
    features["duration"] = pkt_time

    
    features["src_bytes"] = len(pkt[Raw].load) if pkt.haslayer(Raw) else 0
    features["dst_bytes"] = 0  # Direction unknown

    try:
        features["land"] = int(pkt[IP].src == pkt[IP].dst and pkt.sport == pkt.dport)
    except:
        features["land"] = 0

    
    protocol = get_protocol(pkt)
    for proto in ["icmp", "tcp", "udp"]:
        features[f"protocol_type_{proto}"] = int(proto == protocol)

    
    service = get_service(pkt)
    for svc in list(SERVICE_PORTS.values()) + [DEFAULT_SERVICE, "rare"]:
        features[f"service_{svc}"] = int(service == svc)

    
    flag = get_flag(pkt)
    for flag_code in ["S0", "S1", "SF", "S2", "REJ", "RSTR", "SH", "OTH"]:
        features[f"flag_{flag_code}"] = int(flag == flag_code)

    
    if pkt.haslayer(TCP):
        features["tcp_sport"] = pkt[TCP].sport
        features["tcp_dport"] = pkt[TCP].dport
        features["tcp_window"] = pkt[TCP].window
        flags = pkt[TCP].flags
        features["tcp_flag_FIN"] = int(flags & 0x01 != 0)
        features["tcp_flag_SYN"] = int(flags & 0x02 != 0)
        features["tcp_flag_RST"] = int(flags & 0x04 != 0)
        features["tcp_flag_PSH"] = int(flags & 0x08 != 0)
        features["tcp_flag_ACK"] = int(flags & 0x10 != 0)
        features["tcp_flag_URG"] = int(flags & 0x20 != 0)
        features["tcp_flag_ECE"] = int(flags & 0x40 != 0)
        features["tcp_flag_CWR"] = int(flags & 0x80 != 0)
    else:
        features["tcp_sport"] = 0
        features["tcp_dport"] = 0
        features["tcp_window"] = 0
        for bit in ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"]:
            features[f"tcp_flag_{bit}"] = 0


    if pkt.haslayer(UDP):
        features["udp_sport"] = pkt[UDP].sport
        features["udp_dport"] = pkt[UDP].dport
        features["udp_len"] = pkt[UDP].len
    else:
        features["udp_sport"] = 0
        features["udp_dport"] = 0
        features["udp_len"] = 0

    if pkt.haslayer(ICMP):
        features["icmp_type"] = pkt[ICMP].type
        features["icmp_code"] = pkt[ICMP].code
    else:
        features["icmp_type"] = 0
        features["icmp_code"] = 0

   
    flow_data = flow_tracker[flow_id]
    flow_data["timestamps"].append(pkt_time)
    flow_data["packet_count"] += 1

    timestamps = flow_data["timestamps"]
    inter_arrival_times = [
        t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])
    ]

    features["flow_packet_count"] = flow_data["packet_count"]
    features["flow_duration"] = (
        timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    )
    features["packet_rate"] = (
        flow_data["packet_count"] / features["flow_duration"]
        if features["flow_duration"] > 0
        else 0
    )

    
    payload = pkt[Raw].load if pkt.haslayer(Raw) else b""
    features["payload_entropy"] = calculate_entropy(payload)

        
    MAX_FLOW_DURATION = 60  
    if features["flow_duration"] > MAX_FLOW_DURATION:
        flow_tracker[flow_id] = {"timestamps": [], "packet_count": 0}

    
    if predict_label_fn:
        features["prediction"] = predict_label_fn(features)

    
    if save_features:
        save_features_to_csv(features, CSV_FILENAME)
    
       
    return features



def packet_callback(pkt, features_list, label):
    if not pkt.haslayer(IP):
        return

    ip = pkt[IP]
    proto = get_protocol(pkt)
    sport = pkt.sport if hasattr(pkt, "sport") else 0
    dport = pkt.dport if hasattr(pkt, "dport") else 0
    flow_id = f"{ip.src}-{ip.dst}-{proto}-{sport}-{dport}"

    features = extract_features(pkt, flow_tracker, flow_id)
    features["label"] = label
    features_list.append(features)



def main():
    parser = argparse.ArgumentParser(description="Packet Feature Extractor")
    parser.add_argument("--count", type=int, default=100, help="Number of packets to capture")
    parser.add_argument("--output", type=str, default="features.csv", help="Output CSV file")
    parser.add_argument("--label", type=str, choices=["normal", "attack"], default="normal", help="Label for the captured packets")
    args = parser.parse_args()

    features_list = []
    sniff(count=args.count, prn=lambda pkt: packet_callback(pkt, features_list, args.label), store=False)

    if not features_list:
        print("No features to save.")
        return

    
    keys = features_list[0].keys()
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(features_list)

    print(f"Saved {len(features_list)} packets to {args.output}")


if __name__ == "__main__":
    main()
