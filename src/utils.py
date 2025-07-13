import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')


def load_scaler(filename = 'scaler.pkl'):
    path = os.path.join(ARTIFACTS_DIR, filename)
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from {path}")
    return scaler


def load_label_encoder(filename = 'label_encoder.pkl'):
    path = os.path.join(ARTIFACTS_DIR, filename)
    print(f"Loaded the label encoder from {path}")
    return joblib.load(path)



    