import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN, SMOTE
from scipy.stats import randint, uniform
from data_utils import train_test_split, plot_leaderboard

warnings.filterwarnings('ignore')

DATA_PATH = '../data/processed/processed_kdd.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_processed_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_features_target(df: pd.DataFrame, target_col='label'):
    x = df.drop(columns=[target_col])
    y = df[target_col]
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

def train_and_evaluate(model, param_dist, x_train, y_train, x_test, y_test, model_name):
    print(f"\n--- Tuning and Training {model_name} ---")


    class_counts = y_train.value_counts()
    valid_classes = class_counts[class_counts > 1].index
    dropped_classes = set(y_train.unique()) - set(valid_classes)
    if dropped_classes:
        print(f"[!] Dropping {len(dropped_classes)} ultra-minority classes from training (less than 2 samples)")

    
    train_mask = y_train.isin(valid_classes)
    test_mask = y_test.isin(valid_classes)
    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    print(f"[i] Classes used for {model_name}: {sorted(y_train.unique())}")

    min_class_count = y_train.value_counts().min()
    k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
    resampled = False


    if model_name == "logistic_regression":
        print("[i] Skipping SMOTE/ADASYN for logistic regression. Using class_weight='balanced'.")
        model.set_params(class_weight='balanced')
    else:
        try:
            print("[+] Applying SMOTE oversampling...")
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            x_train, y_train = smote.fit_resample(x_train, y_train)
            resampled = True
        except Exception as e:
            print(f"[!] SMOTE failed: {e}")

        if not resampled:
            try:
                print("[+] Applying ADASYN oversampling...")
                adasyn = ADASYN(random_state=42, n_neighbors=k_neighbors)
                x_train, y_train = adasyn.fit_resample(x_train, y_train)
                resampled = True
            except Exception as e:
                print(f"[!] ADASYN failed: {e}")

        if not resampled and hasattr(model, 'class_weight'):
            print("[!] Falling back to class weights without resampling.")
            model.set_params(class_weight='balanced')


    cv_folds = 3 if model_name == "logistic_regression" else 5
    parallel_jobs = 1 if model_name == "logistic_regression" else 2

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        scoring='f1_macro',
        cv=cv_folds,
        n_iter=10,
        verbose=1,
        n_jobs=parallel_jobs
    )
    search.fit(x_train, y_train)

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
    print(f" Leaderboard updated at {leaderboard_path}")
from sklearn.preprocessing import LabelEncoder

def train_and_evaluate(model, param_dist, x_train, y_train, x_test, y_test, model_name):
    print(f"\n--- Tuning and Training {model_name} ---")

    # Drop ultra-minority classes
    class_counts = y_train.value_counts()
    valid_classes = class_counts[class_counts > 1].index
    dropped_classes = set(y_train.unique()) - set(valid_classes)
    if dropped_classes:
        print(f"[!] Dropping {len(dropped_classes)} ultra-minority classes from training (less than 2 samples)")
        mask = y_train.isin(valid_classes)
        x_train, y_train = x_train[mask], y_train[mask]

    # Label encoding (only for models that need contiguous labels)
    label_encoder = None
    y_train_encoded, y_test_encoded = y_train, y_test
    if model_name == "xgboost":
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        print(f"[i] Classes used for {model_name}: {label_encoder.classes_}")
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test

    # Handle class imbalance
    min_class_count = pd.Series(y_train_encoded).value_counts().min()
    k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
    resampled = False

    if model_name == "logistic_regression":
        print("[i] Skipping SMOTE/ADASYN for logistic regression. Using class_weight='balanced'.")
        model.set_params(class_weight='balanced')
    else:
        try:
            print(f"[+] Applying SMOTE oversampling...")
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            x_train, y_train_encoded = smote.fit_resample(x_train, y_train_encoded)
            resampled = True
        except Exception as e:
            print(f"[!] SMOTE failed: {e}")

        if not resampled:
            try:
                print(f"[+] Applying ADASYN oversampling...")
                adasyn = ADASYN(random_state=42, n_neighbors=k_neighbors)
                x_train, y_train_encoded = adasyn.fit_resample(x_train, y_train_encoded)
                resampled = True
            except Exception as e:
                print(f"[!] ADASYN failed: {e}")

        if not resampled and hasattr(model, 'class_weight'):
            print("[!] Falling back to class weights without resampling.")
            model.set_params(class_weight='balanced')

    # Search params
    cv_folds = 3 if model_name == "logistic_regression" else 5
    parallel_jobs = 1 if model_name == "logistic_regression" else 2

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        scoring='f1_macro',
        cv=cv_folds,
        n_iter=10,
        verbose=1,
        n_jobs=parallel_jobs,
        error_score='raise'  # Optional: raises helpful errors early
    )
    search.fit(x_train, y_train_encoded)

    best_model = search.best_estimator_
    preds_encoded = best_model.predict(x_test)

    # Decode predictions if label-encoded
    if label_encoder:
        preds = label_encoder.inverse_transform(preds_encoded)
    else:
        preds = preds_encoded

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)

    print(f"Best Params: {search.best_params_}")
    print(f"Best CV F1 Score: {search.best_score_:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(report)

    # Save report
    report_path = os.path.join(MODEL_DIR, f"{model_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Best Params: {search.best_params_}\n")
        f.write(f"Best CV F1 Score: {search.best_score_:.4f}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(report)

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    joblib.dump(best_model, model_path)
    print(f"{model_name} saved to {model_path}")

    # Save confusion matrix
    ConfusionMatrixDisplay.from_estimator(best_model, x_test, y_test, cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(os.path.join(MODEL_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # Update leaderboard
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
    print(f" Leaderboard updated at {leaderboard_path}")

def main():
    df = load_processed_data(DATA_PATH)
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # ✅ Run one model at a time for now
    model_name = "knn"  # change to other model names as needed
    cfg = MODEL_PARAMS[model_name]
    train_and_evaluate(cfg["model"], cfg["params"], x_train, y_train, x_test, y_test, model_name)

    plot_leaderboard()

if __name__ == "__main__":
    main()
