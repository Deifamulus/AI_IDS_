import pandas as pd
import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_utils import train_test_split, plot_leaderboard


DATA_PATH = '../data/processed/processed_kdd.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


def load_processed_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_features_target(df: pd.DataFrame, target_col='label'):
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y


def safe_smote(x, y, random_state=42):
    """Apply SMOTE only to classes with more than 1 sample."""
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1

    print(f"Using SMOTE with k_neighbors={k_neighbors}")
    try:
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        x_resampled, y_resampled = smote.fit_resample(x, y)
        return x_resampled, y_resampled
    except ValueError as e:
        print(f"[!] SMOTE failed: {e}")
        return x, y  


MODEL_PARAMS = {
    "random_forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": randint(50, 200),
            "max_depth": randint(3, 20),
            "min_samples_split": randint(2, 10)
        }
    },
    "svm": {
        "model": SVC(),
        "params": {
            "C": uniform(0.1, 10),
            "kernel": ["linear", "rbf"],
            "gamma": ['scale', 'auto']
        }
    },
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": uniform(0.1, 10),
            "solver": ["liblinear", "lbfgs"]
        }
    },
    "xgboost": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
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


def train_and_evaluate(model, param_dist, x_train, y_train, x_test, y_test, model_name):
    print(f"\n--- Tuning and Training {model_name} ---")

    
    if hasattr(model, 'class_weight'):
        model.set_params(class_weight='balanced')

    
    x_train_resampled, y_train_resampled = safe_smote(x_train, y_train)

    
    search = RandomizedSearchCV(model, param_distributions=param_dist, scoring='f1_macro',
                                cv=5, n_iter=10, verbose=1, n_jobs=-1)
    search.fit(x_train_resampled, y_train_resampled)
    best_model = search.best_estimator_

    
    preds = best_model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)

    print(f"Best Params: {search.best_params_}")
    print(f"Best CV F1 Score: {search.best_score_:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(report)

    
    report_path = os.path.join(MODEL_DIR, f"{model_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Best Params: {search.best_params_}\n")
        f.write(f"Best CV F1 Score: {search.best_score_:.4f}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(report)

    
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    joblib.dump(best_model, model_path)
    print(f"{model_name} saved to {model_path}")

    
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
    print(f"📊 Leaderboard updated at {leaderboard_path}")


def main():
    df = load_processed_data(DATA_PATH)
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    for name, cfg in MODEL_PARAMS.items():
        train_and_evaluate(cfg["model"], cfg["params"], x_train, y_train, x_test, y_test, name)

    plot_leaderboard()

if __name__ == "__main__":
    main()
