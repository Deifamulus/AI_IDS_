import os
import joblib
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN, SMOTE
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from data_utils import train_test_split, plot_leaderboard
import json 

warnings.filterwarnings("ignore")

DATA_PATH = '../data/processed/processed_kdd.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PARAMS = {
    "random_forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": randint(50, 200),
            "max_depth": randint(3, 20),
            "min_samples_split": randint(2, 10)
        }
    },
    "logistic_regression": {
        "model": LogisticRegression(max_iter=5000, n_jobs=1, warm_start=True),
        "params": {
            "C": uniform(0.1, 10),
            "solver": ['saga']
        }
    },
    "xgboost": {
        "model": XGBClassifier(eval_metric='mlogloss'),
        "params": {
            "n_estimators": randint(50, 150),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.3)
        }
    },
    "knn": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": randint(3, 15),
            "weights": ["uniform", "distance"]
        }
    }
}

def load_processed_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_features_target(df: pd.DataFrame, target_col='label'):
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y

def train_and_evaluate(model, param_dist, x_train, y_train, x_test, y_test, model_name):
    print(f"\n--- Tuning and Training {model_name} ---")

    class_counts = y_train.value_counts()
    valid_classes = class_counts[class_counts > 1].index
    x_train, y_train = x_train[y_train.isin(valid_classes)], y_train[y_train.isin(valid_classes)]
    x_test, y_test = x_test[y_test.isin(valid_classes)], y_test[y_test.isin(valid_classes)]

    label_encoder = None
    y_train_encoded, y_test_encoded = y_train, y_test

    if model_name == "xgboost":
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        print(f"[i] Classes used for {model_name}: {label_encoder.classes_}")

        
        joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder_xgboost.pkl"))
        print(f"[✓] Label encoder saved to {os.path.join(MODEL_DIR, 'label_encoder_xgboost.pkl')}")

    min_class_count = pd.Series(y_train_encoded).value_counts().min()
    k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
    resampled = False

    if model_name == "logistic_regression":
        print("[i] Skipping SMOTE for logistic regression, using class_weight='balanced'.")
        model.set_params(class_weight='balanced')
    else:
        try:
            print("[+] Applying SMOTE...")
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            x_train, y_train_encoded = smote.fit_resample(x_train, y_train_encoded)
            resampled = True
        except Exception as e:
            print(f"[!] SMOTE failed: {e}")
            try:
                print("[+] Trying ADASYN instead...")
                adasyn = ADASYN(random_state=42, n_neighbors=k_neighbors)
                x_train, y_train_encoded = adasyn.fit_resample(x_train, y_train_encoded)
                resampled = True
            except Exception as e:
                print(f"[!] ADASYN also failed: {e}")

        if not resampled and hasattr(model, 'class_weight'):
            print("[!] Falling back to class weights.")
            model.set_params(class_weight='balanced')

    class TQDMSearchCV(RandomizedSearchCV):
        def fit(self, X, y=None, **fit_params):
            with tqdm(total=self.n_iter, desc=f"Searching {model_name}", ncols=100) as pbar:
                self._pbar = pbar
                return super().fit(X, y, **fit_params)

        def _run_search(self, evaluate_candidates):
            def wrapped(candidates):
                results = evaluate_candidates(candidates)
                self._pbar.update(len(candidates))
                return results
            super()._run_search(wrapped)

    search = TQDMSearchCV(
        model,
        param_distributions=param_dist,
        scoring='f1_macro',
        cv=5,
        n_iter=10,
        verbose=0,
        n_jobs=2
    )
    search.fit(x_train, y_train_encoded)

    best_model = search.best_estimator_
    preds_encoded = best_model.predict(x_test)
    preds = label_encoder.inverse_transform(preds_encoded) if label_encoder else preds_encoded

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)

    print(f"\nBest Params: {search.best_params_}")
    print(f"Best CV F1 Score: {search.best_score_:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(report)

    report_path = os.path.join(MODEL_DIR, f"{model_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Best Params: {search.best_params_}\n")
        f.write(f"Best CV F1 Score: {search.best_score_:.4f}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(report)

    model_filename = "xgboost_model_final.pkl" if model_name == "xgboost" else f"{model_name}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    joblib.dump(best_model, model_path)
    print(f"[✓] {model_name} model saved to {model_path}")

    ConfusionMatrixDisplay.from_estimator(best_model, x_test, y_test, cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(os.path.join(MODEL_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()

    leaderboard_path = os.path.join(MODEL_DIR, "leaderboard.csv")
    result = {
        "Model": model_name,
        "Best_F1_CV": search.best_score_,
        "Test_Accuracy": acc,
        "Best_Params": str(search.best_params_)
    }

    if os.path.exists(leaderboard_path):
        leaderboard_df = pd.read_csv(leaderboard_path)
        leaderboard_df = pd.concat([leaderboard_df, pd.DataFrame([result])], ignore_index=True)
    else:
        leaderboard_df = pd.DataFrame([result])

    leaderboard_df.to_csv(leaderboard_path, index=False)
    print(f"[✓] Leaderboard updated at {leaderboard_path}")

def main():
    df = load_processed_data(DATA_PATH)
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    model_name = "xgboost"  # Change this to try other models
    cfg = MODEL_PARAMS[model_name]
    train_and_evaluate(cfg["model"], cfg["params"], x_train, y_train, x_test, y_test, model_name)

    plot_leaderboard()

if __name__ == "__main__":
    main()
