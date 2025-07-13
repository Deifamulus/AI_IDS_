import joblib

def load_scaler(scaler_path : str):
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from {scaler_path}")
    return scaler


def load_label_encoder(path = 'artifacts/label_encoder.pkl'):
    print(f"Loaded the label encoder from {path}")
    return joblib.load(path)



    