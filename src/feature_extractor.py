import pandas as pd
from scapy.all import sniff, Raw
from scapy.layers.inet import IP, TCP, UDP, ICMP
import argparse
import datetime
import os


features_list = []


SERVICE_PORTS = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "domain_u", 80: "http", 110: "pop3", 443: "https",
    20: "ftp_data", 79: "finger", 106: "pop3pw", 143: "imap", 194: "irc", 443: "https"
}
DEFAULT_SERVICE = "other"

FLAGS_LIST = ['F', 'S', 'R', 'P', 'A', 'U']


def get_protocol(pkt):
    if pkt.haslayer(IP):
        proto = pkt[IP].proto
        if proto == 1:
            return "icmp"
        elif proto == 6:
            return "tcp"
        elif proto == 17:
            return "udp"
    return "other"


def get_service(pkt):
    if pkt.haslayer(TCP) or pkt.haslayer(UDP):
        dport = pkt[TCP].dport if pkt.haslayer(TCP) else pkt[UDP].dport
        return SERVICE_PORTS.get(dport, DEFAULT_SERVICE)
    return DEFAULT_SERVICE


def get_flag(pkt):
    if pkt.haslayer(TCP):
        flags = pkt[TCP].flags
        flag_str = str(flags)
        return flag_str  
    return "OTH"


def extract_features(pkt):
    feature_dict = {}

    
    feature_dict['duration'] = pkt.time if hasattr(pkt, "time") else 0

    
    feature_dict['src_bytes'] = len(pkt[Raw].load) if pkt.haslayer(Raw) else 0
    feature_dict['dst_bytes'] = 0 

    
    try:
        feature_dict['land'] = int(
            pkt[IP].src == pkt[IP].dst and
            pkt.sport == pkt.dport
        )
    except:
        feature_dict['land'] = 0

   
    protocol = get_protocol(pkt)
    for proto in ["icmp", "tcp", "udp"]:
        feature_dict[f"protocol_type_{proto}"] = int(proto == protocol)

 
    service = get_service(pkt)
    for svc in list(SERVICE_PORTS.values()) + [DEFAULT_SERVICE, "rare"]:
        feature_dict[f"service_{svc}"] = int(svc == service)

   
    flag = get_flag(pkt)
    for flag_code in ["OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "S0", "S1", "S2", "S3", "SF", "SH"]:
        feature_dict[f"flag_{flag_code}"] = int(flag == flag_code)

 
    if pkt.haslayer(TCP):
        feature_dict['tcp_sport'] = pkt[TCP].sport
        feature_dict['tcp_dport'] = pkt[TCP].dport
        feature_dict['tcp_window'] = pkt[TCP].window
    else:
        feature_dict['tcp_sport'] = 0
        feature_dict['tcp_dport'] = 0
        feature_dict['tcp_window'] = 0

    if pkt.haslayer(UDP):
        feature_dict['udp_sport'] = pkt[UDP].sport
        feature_dict['udp_dport'] = pkt[UDP].dport
        feature_dict['udp_len'] = pkt[UDP].len
    else:
        feature_dict['udp_sport'] = 0
        feature_dict['udp_dport'] = 0
        feature_dict['udp_len'] = 0


    if pkt.haslayer(ICMP):
        feature_dict['icmp_type'] = pkt[ICMP].type
        feature_dict['icmp_code'] = pkt[ICMP].code
    else:
        feature_dict['icmp_type'] = 0
        feature_dict['icmp_code'] = 0

    return feature_dict



def process_packet(pkt):
    try:
        if not pkt.haslayer(IP):
            return

        features = extract_features(pkt)
        features_list.append(features)

    except Exception as e:
        print(f"Error processing packet: {e}")


def run_feature_extractor(iface=None, count=50, output="features.csv", pcap_path=None):
    global features_list
    features_list = []  

    try:
        if pcap_path:
            sniff(offline=pcap_path, prn=process_packet, count=count)
        else:
            sniff(iface=iface, prn=process_packet, count=count)
    except KeyboardInterrupt:
        print("\n[INFO] Capture interrupted.")
    except Exception as e:
        print(f"[ERROR] {e}")

    if features_list:
        df = pd.DataFrame(features_list)
        df.fillna(0, inplace=True)
        df.to_csv(output, index=False)
        print(f"[INFO] Features saved to {output}")
        return df
    else:
        print("[WARN] No features extracted.")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Real-time Feature Extractor for AI-IDS")
    parser.add_argument('--iface', type=str, help="Interface for live capture (e.g., eth0)")
    parser.add_argument('--count', type=int, default=50, help="Number of packets to capture")
    parser.add_argument('--output', type=str, default="features.csv", help="Output CSV file")
    parser.add_argument('--read-pcap', type=str, help="Path to .pcap file instead of live sniff")
    args = parser.parse_args()

    run_feature_extractor(
        iface=args.iface,
        count=args.count,
        output=args.output,
        pcap_path=args.read_pcap
    )

if __name__ == "__main__":
    main()
