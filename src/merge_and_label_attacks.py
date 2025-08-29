import os
import pandas as pd


csv_dir = "csv_outputs"
merged_csv_path = "../data/processed/realtime_merged.csv"


os.makedirs(os.path.dirname(merged_csv_path), exist_ok=True)


attack_labels = {
    "neptune": "neptune",
    "smurf": "smurf",
    "teardrop": "teardrop",
    "back": "back",
    "pod": "pod",
    "land": "land",
    "ipsweep": "ipsweep",
    "portsweep": "portsweep"
}


merged_df = pd.DataFrame()
for filename in os.listdir(csv_dir):
    for attack_key in attack_labels:
        if filename.startswith(attack_key) and filename.endswith(".csv"):
            file_path = os.path.join(csv_dir, filename)
            try:
                df = pd.read_csv(file_path)

                
                df["label"] = attack_labels[attack_key]

                
                merged_df = pd.concat([merged_df, df], ignore_index=True)
                print(f"[✓] Added {filename} with label {attack_labels[attack_key]}")
            except Exception as e:
                print(f"[!] Failed to read {filename}: {e}")


if not merged_df.empty:
    merged_df.to_csv(merged_csv_path, index=False)
    print(f" Merged CSV saved to {merged_csv_path}")
else:
    print("[!] No data was merged.")
