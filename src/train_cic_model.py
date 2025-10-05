"""
Unified Model Training Script for CIC-IDS Dataset
This script trains ML models on preprocessed CIC-IDS data
"""

import os
import joblib
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report, 
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE, ADASYN
from scipy.stats import randint, uniform
import json
from datetime import datetime

warnings.filterwarnings("ignore")


class CICModelTrainer:
    """
    Model trainer for CIC-IDS datasets
    Supports multiple ML algorithms with hyperparameter tuning
    """
    
    def __init__(self, data_path: str, model_dir: str = "models/cic_models"):
        """
        Initialize the trainer
        
        Args:
            data_path: Path to preprocessed CIC data
            model_dir: Directory to save trained models
        """
        self.data_path = data_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        
        # Model configurations
        self.model_configs = {
            "random_forest": {
                "model": RandomForestClassifier(random_state=42, n_jobs=-1),
                "params": {
                    "n_estimators": randint(100, 300),
                    "max_depth": randint(10, 30),
                    "min_samples_split": randint(2, 10),
                    "min_samples_leaf": randint(1, 5)
                }
            },
            "xgboost": {
                "model": None,  # Will be imported if available
                "params": {
                    "n_estimators": randint(100, 300),
                    "max_depth": randint(3, 15),
                    "learning_rate": uniform(0.01, 0.3),
                    "subsample": uniform(0.6, 0.4)
                }
            },
            "logistic_regression": {
                "model": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
                "params": {
                    "C": uniform(0.1, 10),
                    "solver": ['lbfgs', 'saga']
                }
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": randint(100, 300),
                    "max_depth": randint(3, 10),
                    "learning_rate": uniform(0.01, 0.2)
                }
            },
            "knn": {
                "model": KNeighborsClassifier(n_jobs=-1),
                "params": {
                    "n_neighbors": randint(3, 20),
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"]
                }
            },
            "decision_tree": {
                "model": DecisionTreeClassifier(random_state=42),
                "params": {
                    "max_depth": randint(5, 30),
                    "min_samples_split": randint(2, 10),
                    "min_samples_leaf": randint(1, 5)
                }
            }
        }
        
        # Try to import XGBoost
        try:
            from xgboost import XGBClassifier
            self.model_configs["xgboost"]["model"] = XGBClassifier(
                eval_metric='mlogloss', 
                random_state=42,
                n_jobs=-1
            )
        except ImportError:
            print("Warning: XGBoost not available. Skipping XGBoost model.")
            del self.model_configs["xgboost"]
    
    def load_data(self):
        """Load preprocessed data"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} samples with {len(self.df.columns)} features")
        
        # Check for label column
        if 'label' not in self.df.columns:
            raise ValueError("Label column not found in dataset")
        
        # Show class distribution
        print("\nClass distribution:")
        print(self.df['label'].value_counts())
        
        return self.df
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42, min_samples: int = 2):
        """
        Split data into train and test sets, ensuring each class has at least min_samples.
        
        Args:
            test_size: Proportion of the dataset to include in the test split
            random_state: Random seed for reproducibility
            min_samples: Minimum number of samples required per class
        """
        print(f"\nSplitting data (test_size={test_size}, min_samples={min_samples})...")
        
        X = self.df.drop(columns=['label'])
        y = self.df['label']
        
        # Filter out classes with fewer than min_samples
        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts >= min_samples].index
        mask = y.isin(valid_classes)
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        if len(valid_classes) < len(class_counts):
            print(f"Filtered out {len(class_counts) - len(valid_classes)} classes with <{min_samples} samples")
            print(f"Kept {len(valid_classes)} classes with ≥{min_samples} samples each")
            
            # Print removed classes for debugging
            removed_classes = set(class_counts.index) - set(valid_classes)
            if len(removed_classes) > 0:
                print("Removed classes:", ", ".join(removed_classes))
        
        # Split the filtered data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_filtered, 
            y_filtered, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y_filtered
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print("Class distribution in training set:")
        print(self.y_train.value_counts())
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def apply_smote(self, k_neighbors: int = 5):
        """Apply SMOTE to balance training data"""
        print("\n--- Applying SMOTE ---")
        
        try:
            # Check minimum class count
            min_class_count = self.y_train.value_counts().min()
            k = min(k_neighbors, min_class_count - 1)
            
            if k < 1:
                print("Warning: Not enough samples for SMOTE. Skipping.")
                return
            
            smote = SMOTE(random_state=42, k_neighbors=k)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            
            print(f"After SMOTE: {len(self.X_train)} training samples")
            print("Class distribution after SMOTE:")
            print(pd.Series(self.y_train).value_counts())
            
        except Exception as e:
            print(f"SMOTE failed: {e}")
            print("Continuing without SMOTE...")
    
    def train_model(self, model_name: str, use_smote: bool = True, 
                   tune_hyperparams: bool = True, n_iter: int = 20):
        """
        Train a single model
        
        Args:
            model_name: Name of the model to train
            use_smote: Whether to apply SMOTE
            tune_hyperparams: Whether to tune hyperparameters
            n_iter: Number of iterations for hyperparameter search
        
        Returns:
            Trained model and metrics
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        print("\n" + "=" * 60)
        print(f"Training {model_name.upper()}")
        print("=" * 60)
        
        config = self.model_configs[model_name]
        model = config["model"]
        param_dist = config["params"]
        
        # Prepare training data
        X_train, y_train = self.X_train.copy(), self.y_train.copy()
        
        # Apply SMOTE if requested
        if use_smote and model_name != "logistic_regression":
            try:
                min_class_count = pd.Series(y_train).value_counts().min()
                k = min(5, min_class_count - 1)
                
                if k >= 1:
                    smote = SMOTE(random_state=42, k_neighbors=k)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    print(f"Applied SMOTE: {len(X_train)} training samples")
            except Exception as e:
                print(f"SMOTE failed: {e}")
        
        # Hyperparameter tuning
        if tune_hyperparams and param_dist:
            print(f"\nTuning hyperparameters ({n_iter} iterations)...")
            
            search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring='f1_macro',
                cv=5,
                verbose=0,
                n_jobs=-1,
                random_state=42
            )
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            print(f"Best parameters: {search.best_params_}")
            print(f"Best CV F1 score: {search.best_score_:.4f}")
        else:
            print("\nTraining with default parameters...")
            best_model = model
            best_model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = best_model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            "model_name": model_name,
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average='macro', zero_division=0),
            "recall": recall_score(self.y_test, y_pred, average='macro', zero_division=0),
            "f1_score": f1_score(self.y_test, y_pred, average='macro', zero_division=0)
        }
        
        # Print metrics
        print("\n--- Test Set Performance ---")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        
        # Classification report
        print("\n--- Classification Report ---")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        
        # Save model
        model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"\n✓ Model saved to {model_path}")
        
        # Save metrics
        metrics_path = os.path.join(self.model_dir, f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(self.y_test, y_pred, model_name)
        
        return best_model, metrics
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = os.path.join(self.model_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to {cm_path}")
    
    def train_all_models(self, use_smote: bool = True, tune_hyperparams: bool = True):
        """Train all available models"""
        results = []
        
        print("\n" + "=" * 60)
        print("TRAINING ALL MODELS")
        print("=" * 60)
        
        for model_name in self.model_configs.keys():
            try:
                model, metrics = self.train_model(
                    model_name, 
                    use_smote=use_smote, 
                    tune_hyperparams=tune_hyperparams
                )
                results.append(metrics)
            except Exception as e:
                print(f"\n✗ Error training {model_name}: {e}")
                continue
        
        # Create leaderboard
        self._create_leaderboard(results)
        
        return results
    
    def _create_leaderboard(self, results: list):
        """Create and save model leaderboard"""
        if not results:
            print("No results to create leaderboard")
            return
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('f1_score', ascending=False)
        
        # Save leaderboard
        leaderboard_path = os.path.join(self.model_dir, "leaderboard.csv")
        df_results.to_csv(leaderboard_path, index=False)
        
        print("\n" + "=" * 60)
        print("MODEL LEADERBOARD")
        print("=" * 60)
        print(df_results.to_string(index=False))
        print(f"\n✓ Leaderboard saved to {leaderboard_path}")
        
        # Plot leaderboard
        self._plot_leaderboard(df_results)
    
    def _plot_leaderboard(self, df_results: pd.DataFrame):
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            df_sorted = df_results.sort_values(metric)
            ax.barh(df_sorted['model_name'], df_sorted[metric], color='skyblue')
            ax.set_xlabel(title)
            ax.set_title(f'{title} Comparison')
            ax.set_xlim([0, 1])
            
            # Add value labels
            for i, v in enumerate(df_sorted[metric]):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plot_path = os.path.join(self.model_dir, "model_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison plot saved to {plot_path}")


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ML models on CIC-IDS dataset")
    parser.add_argument("--data", type=str, required=True, 
                       help="Path to preprocessed CIC data")
    parser.add_argument("--model-dir", type=str, default="models/cic_models",
                       help="Directory to save models")
    parser.add_argument("--model", type=str, choices=[
        "random_forest", "xgboost", "logistic_regression", 
        "gradient_boosting", "knn", "decision_tree", "all"
    ], default="all", help="Model to train")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE")
    parser.add_argument("--no-tune", action="store_true", help="Disable hyperparameter tuning")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--n-iter", type=int, default=20, help="Hyperparameter search iterations")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CICModelTrainer(data_path=args.data, model_dir=args.model_dir)
    
    # Load and split data
    trainer.load_data()
    trainer.split_data(test_size=args.test_size)
    
    # Train models
    use_smote = not args.no_smote
    tune_hyperparams = not args.no_tune
    
    if args.model == "all":
        trainer.train_all_models(use_smote=use_smote, tune_hyperparams=tune_hyperparams)
    else:
        trainer.train_model(args.model, use_smote=use_smote, 
                          tune_hyperparams=tune_hyperparams, n_iter=args.n_iter)
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
