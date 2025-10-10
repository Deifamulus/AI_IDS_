"""
Unified Model Training Script for CIC-IDS Dataset
This script trains ML models on preprocessed CIC-IDS data
"""

import os
import gc
import time
import psutil
import joblib
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from datetime import datetime
from pathlib import Path
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Set pandas to use more efficient data types by default
pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning
pd.set_option('display.precision', 4)
pd.set_option('display.max_columns', 100)

# Memory tracking
def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Convert to MB

class MemoryTracker:
    """Context manager to track memory usage"""
    def __enter__(self):
        self.start_mem = get_memory_usage()
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_mem = get_memory_usage()
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.mem_used = self.end_mem - self.start_mem
        print(f"Memory used: {self.mem_used:.2f} MB | Duration: {self.duration:.2f}s")

# Custom tqdm callback for model training
class TQDMCallback:
    def __init__(self, desc=None, total=None):
        self.pbar = tqdm(desc=desc, total=total, unit='epoch', dynamic_ncols=True)
        self.current_epoch = 0
    
    def __call__(self, env):
        self.current_epoch += 1
        self.pbar.update(1)
        metrics = {k: v for k, v in env.evaluation_result_list if not k.startswith('val_')}
        self.pbar.set_postfix(metrics)
    
    def close(self):
        self.pbar.close()
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
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
    make_scorer
)
from sklearn.model_selection import (
    train_test_split, 
    RandomizedSearchCV, 
    cross_validate,
    StratifiedKFold,
    cross_val_score
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE, ADASYN
from scipy.stats import randint, uniform
import json
from datetime import datetime

warnings.filterwarnings("ignore")


class CICModelTrainer:
    """
    Memory-efficient model trainer for CIC-IDS datasets with progress tracking
    
    Features:
    - Memory-efficient data loading with chunking
    - Progress tracking with tqdm
    - GPU acceleration support
    - Model checkpointing
    - Resource monitoring
    """
    
    def __init__(self, data_path: str, model_dir: str = "models/cic_models", 
                 feature_mapping_path: str = None, use_gpu: bool = False):
        """
        Initialize the trainer with memory-efficient settings
        
        Args:
            data_path: Path to preprocessed CIC data
            model_dir: Directory to save trained models
            feature_mapping_path: Optional path to feature mapping JSON
            use_gpu: Whether to enable GPU acceleration if available
        """
        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.feature_mapping_path = feature_mapping_path
        self.use_gpu = use_gpu
        self.device = None
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize attributes
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        self.class_weights = None
        self.feature_mapping = {}
        
        # Memory tracking
        self.memory_usage = []
        self.training_start_time = None
        
        # Load feature mapping if provided
        if feature_mapping_path and Path(feature_mapping_path).exists():
            with open(feature_mapping_path, 'r') as f:
                self.feature_mapping = json.load(f)
        
        # Initialize model configurations
        self._initialize_model_configs()
        
        # Enable GPU if requested and available
        if self.use_gpu:
            self._setup_gpu()
    
    def _setup_gpu(self):
        """Configure GPU acceleration if available"""
        try:
            import torch
            if torch.cuda.is_available():
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                self.device = torch.device('cuda')
                
                # Configure XGBoost to use GPU if available
                if 'xgboost' in self.model_configs:
                    if 'params' not in self.model_configs['xgboost']:
                        self.model_configs['xgboost']['params'] = {}
                    self.model_configs['xgboost']['params']['tree_method'] = 'gpu_hist'
                    self.model_configs['xgboost']['params']['gpu_id'] = 0
                    print("Enabled GPU acceleration for XGBoost")
                
                return True
        except Exception as e:
            print(f"Error setting up GPU: {e}")
        
        print("GPU not available or failed to initialize. Using CPU.")
        self.device = None
        return False
    
    def _initialize_model_configs(self, n_estimators=100, max_depth=None, max_features=0.5, 
                                 min_samples_split=2, min_samples_leaf=1, learning_rate=0.1,
                                 min_child_weight=1, gamma=0, subsample=1.0, colsample_bytree=1.0):
        """Initialize model configurations with their hyperparameter spaces
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            max_features: Fraction of features to consider at each split
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            learning_rate: XGBoost learning rate (default: 0.1)
            min_child_weight: XGBoost minimum child weight (default: 1)
            gamma: XGBoost gamma (default: 0)
            subsample: XGBoost subsample ratio (default: 1.0)
            colsample_bytree: XGBoost column subsample ratio (default: 1.0)
        """
        # Calculate class weights for imbalanced data
        class_weights = None
        if self.y_train is not None:
            class_counts = np.bincount(self.y_train)
            class_weights = {i: sum(class_counts) / (len(class_counts) * count) 
                           for i, count in enumerate(class_counts)}
        
        self.model_configs = {
            "xgboost": {
                "model": xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth or 6,
                    learning_rate=learning_rate,
                    min_child_weight=min_child_weight,
                    gamma=gamma,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    objective='multi:softprob',
                    random_state=42,
                    n_jobs=-1,
                    tree_method='hist',  # More memory efficient
                    enable_categorical=False,
                    use_label_encoder=False,
                    scale_pos_weight=1.0,  # Will be updated during training
                    eval_metric='mlogloss',
                    early_stopping_rounds=10,
                    verbosity=1
                ),
                "param_grid": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'min_child_weight': [1, 3, 5],
                    'gamma': [0, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            },
            "random_forest": {
                "model": RandomForestClassifier(
                    n_estimators=min(10, n_estimators),  # Start with fewer trees
                    max_depth=min(6, max_depth) if max_depth else 6,  # Shallower trees
                    max_features=0.1,  # Fewer features per split
                    min_samples_split=max(100, min_samples_split * 5),  # Larger min samples
                    min_samples_leaf=max(50, min_samples_leaf * 5),  # Larger min leaf size
                    n_jobs=-1,
                    random_state=42,
                    verbose=1,
                    warm_start=True,  # Enable incremental fitting
                    max_samples=0.1,  # Use only 10% of data per tree
                    bootstrap=True,
                    class_weight='balanced'
                ),
                "params": {
                    "n_estimators": randint(max(50, n_estimators//2), n_estimators*2),
                    "max_depth": randint(5, 30) if max_depth is None else [max_depth],
                    "min_samples_split": randint(2, 10),
                    "min_samples_leaf": randint(1, 5),
                    "max_features": [max_features],
                    "class_weight": ['balanced', 'balanced_subsample']
                }
            },
            "xgboost": {
                "model": None,  # Will be imported if available
                "params": {
                    "n_estimators": randint(100, 500),
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
    
    def _get_optimized_dtypes(self, df_sample):
        """Determine optimal data types for memory efficiency"""
        dtypes = {}
        
        for col in df_sample.columns:
            col_type = df_sample[col].dtype
            
            if col_type == 'float64':
                # Downcast float64 to float32 for memory efficiency
                dtypes[col] = 'float32'
            elif col_type == 'int64':
                # Downcast int64 to smallest possible int type
                col_min = df_sample[col].min()
                col_max = df_sample[col].max()
                
                if col_min >= 0:  # Unsigned int
                    if col_max < 2**8:
                        dtypes[col] = 'uint8'
                    elif col_max < 2**16:
                        dtypes[col] = 'uint16'
                    elif col_max < 2**32:
                        dtypes[col] = 'uint32'
                else:  # Signed int
                    if col_min > -2**7 and col_max < 2**7:
                        dtypes[col] = 'int8'
                    elif col_min > -2**15 and col_max < 2**15:
                        dtypes[col] = 'int16'
                    elif col_min > -2**31 and col_max < 2**31:
                        dtypes[col] = 'int32'
        
        return dtypes
    
    def load_data(self, chunksize=100000, n_estimators=100, max_depth=None, max_features=0.5, 
                  min_samples_split=2, min_samples_leaf=1):
        """
        Load preprocessed data in chunks with memory optimization
        
        Args:
            chunksize: Number of rows to process at a time
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            max_features: Fraction of features to consider at each split
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            
        Returns:
            pd.DataFrame: Combined and optimized dataset
        """
        print(f"Loading data from {self.data_path} in chunks...")
        
        # First pass: determine optimal dtypes and get class distribution
        print("Analyzing data structure...")
        sample = pd.read_csv(self.data_path, nrows=1000)
        dtypes = self._get_optimized_dtypes(sample)
        
        # Ensure label column is present
        if 'label' not in sample.columns:
            raise ValueError("Label column 'label' not found in dataset")
        
        # Process data in chunks
        chunks = []
        total_rows = 0
        
        # Get total number of rows for progress tracking
        total_rows = sum(1 for _ in open(self.data_path)) - 1  # Subtract header
        print(f"Processing {total_rows:,} rows in chunks of {chunksize:,}...")
        
        # Process chunks
        chunk_iterator = pd.read_csv(
            self.data_path,
            dtype=dtypes,
            chunksize=chunksize,
            low_memory=False
        )
        
        # Process each chunk
        for i, chunk in enumerate(chunk_iterator):
            # Apply feature name mapping if available
            if self.feature_mapping:
                feature_columns = [col for col in chunk.columns if col != 'label']
                new_columns = {}
                for col in feature_columns:
                    if col in self.feature_mapping:
                        new_columns[col] = self.feature_mapping[col]
                    elif col.replace('feature_', '') in self.feature_mapping:
                        new_columns[col] = self.feature_mapping[col.replace('feature_', '')]
                
                if new_columns:
                    chunk = chunk.rename(columns=new_columns)
            
            # Convert dtypes to optimal ones
            for col, dtype in dtypes.items():
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(dtype)
            
            chunks.append(chunk)
            
            # Print progress
            processed = min((i + 1) * chunksize, total_rows)
            print(f"\rProcessed {processed:,}/{total_rows:,} rows "
                  f"({processed/total_rows:.1%})", end="")
        
        print("\nCombining chunks...")
        self.df = pd.concat(chunks, ignore_index=True)
        
        print(f"Loaded {len(self.df)} samples with {len(self.df.columns) - 1} features")
        
        # Show class distribution
        print("\nClass distribution:")
        class_dist = self.df['label'].value_counts()
        print(class_dist)
        # Calculate class weights for later use
        self.class_weights = dict((i, len(self.df) / (len(class_dist) * count)) 
                                for i, count in enumerate(class_dist))
        
        # Initialize model configurations with class weights and custom parameters
        self._initialize_model_configs(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        
        # Clean up
        del chunks
        import gc
        gc.collect()
        
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
    
    def _apply_adasyn(self, sampling_strategy='auto', n_neighbors=5, random_state=42):
        """Apply ADASYN to balance training data"""
        print("\n--- Applying ADASYN ---")
        
        try:
            from imblearn.over_sampling import ADASYN
            
            # Check minimum class count
            min_class_count = self.y_train.value_counts().min()
            k = min(n_neighbors, min_class_count - 1)
            
            if k < 1:
                print("Not enough samples in minority class for ADASYN. Using SMOTE instead.")
                return self.apply_smote()
            
            print(f"Before ADASYN - Class distribution: \n{self.y_train.value_counts()}")
            
            adasyn = ADASYN(
                sampling_strategy=sampling_strategy,
                n_neighbors=k,
                random_state=random_state,
                n_jobs=-1
            )
            
            self.X_train, self.y_train = adasyn.fit_resample(self.X_train, self.y_train)
            
            print(f"After ADASYN - Class distribution: \n{pd.Series(self.y_train).value_counts()}")
            
        except ImportError:
            print("imbalanced-learn not installed. Using SMOTE instead.")
            self.apply_smote()
        except Exception as e:
            print(f"Error applying ADASYN: {e}")
            print("Continuing with imbalanced data...")
    
    def apply_smote(self, k_neighbors: int = 5, random_state: int = 42):
        """
        Apply SMOTE to balance training data
        
        Args:
            k_neighbors: Number of nearest neighbors to use for SMOTE
            random_state: Random seed for reproducibility
        """
        print("\n--- Applying SMOTE ---")
        
        try:
            from imblearn.over_sampling import SMOTE
            
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
    
    def apply_smote_batch(self, X, y, batch_size=5000):
        """Apply SMOTE in memory-efficient batches
        
        Args:
            X: Features
            y: Target
            batch_size: Number of samples per batch
            
        Returns:
            Resampled X and y
        """
        from imblearn.over_sampling import SMOTE
        from sklearn.utils import resample
        import numpy as np
        
        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)
        
        # Find majority and minority classes
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_classes = [c for c in unique_classes if c != majority_class]
        
        # Process each minority class separately
        X_resampled = []
        y_resampled = []
        
        for target_class in tqdm(minority_classes, desc="SMOTE per class"):
            # Get samples for this class
            class_mask = (y == target_class)
            X_class = X[class_mask]
            y_class = y[class_mask]
            
            # Get majority class samples for this batch
            X_majority = X[y == majority_class]
            
            # Process in batches
            n_batches = (len(X_class) + batch_size - 1) // batch_size
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_class))
                
                X_batch = X_class.iloc[start_idx:end_idx]
                y_batch = y_class.iloc[start_idx:end_idx]
                
                # Sample majority class for this batch
                X_maj_batch = resample(X_majority, 
                                     n_samples=len(X_batch), 
                                     random_state=42)
                
                # Combine with current batch
                X_combined = pd.concat([X_batch, X_maj_batch])
                y_combined = pd.Series([1] * len(X_batch) + [0] * len(X_maj_batch))
                
                # Apply SMOTE
                try:
                    smote = SMOTE(random_state=42, 
                                 k_neighbors=min(5, len(X_batch) - 1))
                    X_res, y_res = smote.fit_resample(X_combined, y_combined)
                    
                    # Only keep the synthetic minority samples
                    X_res = X_res[y_res == 1]
                    y_res = pd.Series([target_class] * len(X_res))
                    
                    X_resampled.append(X_res)
                    y_resampled.append(y_res)
                    
                except Exception as e:
                    print(f"SMOTE failed for class {target_class}: {e}")
                    # Keep original samples if SMOTE fails
                    X_resampled.append(X_batch)
                    y_resampled.append(y_batch)
                
                # Clear memory
                del X_batch, y_batch, X_combined, y_combined
                if 'X_res' in locals():
                    del X_res, y_res
                gc.collect()
        
        # Combine all resampled data
        if X_resampled:
            X_resampled = pd.concat(X_resampled)
            y_resampled = pd.concat(y_resampled)
            
            # Add original minority class samples
            X_final = pd.concat([X, X_resampled])
            y_final = pd.concat([y, y_resampled])
            
            # Shuffle the data
            idx = np.random.permutation(len(X_final))
            return X_final.iloc[idx], y_final.iloc[idx]
        
        return X, y

    def _train_xgboost(self, model, X_train, y_train, X_val=None, y_val=None, batch_size=10000):
        """Train XGBoost model with early stopping and memory efficiency"""
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        # Store class information
        self.classes_ = np.unique(y_train)
        
        # Convert to DMatrix - more efficient for XGBoost
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        # Calculate class weights for imbalanced dataset
        class_weights = len(y_train) / (len(self.classes_) * np.bincount(y_train))
        weight = np.array([class_weights[c] for c in y_train])
        
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weight)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set up parameters
        params = model.get_params()
        params['num_class'] = len(self.classes_)
        
        # Train in batches if needed
        if len(X_train) > batch_size * 2:  # Only batch if dataset is large
            num_boost_round = params.pop('n_estimators', 100)
            
            # Initialize model
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, 'eval')],
                early_stopping_rounds=params.get('early_stopping_rounds', 10),
                verbose_eval=10
            )
        else:
            # Train on full dataset
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params.pop('n_estimators', 100),
                evals=[(dval, 'eval')],
                early_stopping_rounds=params.get('early_stopping_rounds', 10),
                verbose_eval=10
            )
        
        # Create a classifier with the trained booster
        model = xgb.XGBClassifier()
        model._Booster = model
        model._le = LabelEncoder().fit(y_train)
        model.classes_ = self.classes_
        
        return model

    def train_model(self, model_name: str, use_smote: bool = True, 
                   tune_hyperparams: bool = False, n_iter: int = 5,
                   smote_batch_size: int = 5000, train_batch_size: int = 5000):
        """
        Train a single model with memory-efficient processing
        
        Args:
            model_name: Name of the model to train
            use_smote: Whether to apply SMOTE
            tune_hyperparams: Whether to tune hyperparameters
            n_iter: Number of iterations for hyperparameter search
            smote_batch_size: Number of samples per SMOTE batch
            train_batch_size: Number of samples per training batch
            
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
        
        # Initialize training start time
        self.training_start_time = time.time()
        
        # Prepare training data
        X_train, y_train = self.X_train.copy(), self.y_train.copy()
        
        # Apply SMOTE in batches if requested
        if use_smote and model_name != "logistic_regression":
            try:
                print("\nApplying SMOTE in optimized batches...")
                X_train, y_train = self.apply_smote_batch(X_train, y_train, batch_size=smote_batch_size)
                print(f"✓ SMOTE applied. Final training size: {len(X_train):,} samples")
                
                # Show class distribution after SMOTE
                print("\nClass distribution after SMOTE:")
                print(pd.Series(y_train).value_counts())
                
            except Exception as e:
                print(f"SMOTE failed: {e}")
                print("Continuing without SMOTE...")
        
        # Hyperparameter tuning
        if tune_hyperparams and param_dist:
            print(f"\nTuning hyperparameters...")
            
            try:
                # For large datasets, use a subset for hyperparameter tuning
                if len(X_train) > 100000:
                    print("Using subset for hyperparameter tuning...")
                    X_tune, _, y_tune, _ = train_test_split(
                        X_train, y_train, 
                        train_size=100000, 
                        stratify=y_train,
                        random_state=42
                    )
                else:
                    X_tune, y_tune = X_train, y_train
                
                # Reduce n_iter for large datasets
                effective_n_iter = min(n_iter, 10 if len(X_train) > 100000 else n_iter)
                
                search = RandomizedSearchCV(
                    model,
                    param_distributions=param_dist,
                    n_iter=effective_n_iter,
                    scoring='f1_macro',
                    cv=3,  # Fewer folds for speed
                    verbose=1,
                    n_jobs=-1,
                    random_state=42,
                    error_score='raise'
                )
                
                search.fit(X_tune, y_tune)
                best_model = search.best_estimator_
                
                print(f"Best parameters: {search.best_params_}")
                print(f"Best CV F1 score: {search.best_score_:.4f}")
                
                # Clear memory
                if len(X_train) > 100000:
                    del X_tune, y_tune
                    gc.collect()
                
                # Train final model with best parameters on full data in batches
                print("\nTraining final model with best parameters...")
                best_model = best_model.set_params(**search.best_params_)
                
            except Exception as e:
                print(f"Hyperparameter tuning failed: {e}")
                print("Falling back to default parameters...")
                best_model = model
        else:
            print("\nTraining with default parameters...")
            best_model = model
        
        # Store class information for consistent prediction
        self.classes_ = np.unique(y_train)
        
        # Train the model with appropriate method
        try:
            if model_name == 'random_forest':
                # Use specialized batch training for Random Forest
                print("Using specialized Random Forest batch training...")
                best_model = self._train_random_forest_in_batches(
                    best_model,
                    X_train,
                    y_train,
                    batch_size=train_batch_size
                )
                
                # Ensure the model has the correct classes for prediction
                if hasattr(best_model, 'classes_'):
                    best_model.classes_ = self.classes_
            elif hasattr(best_model, 'partial_fit'):
                # For models that support online learning
                print(f"Training in batches of {train_batch_size} samples...")
                classes = np.unique(y_train)
                best_model.fit(X_train.iloc[:1], y_train.iloc[:1])  # Dummy fit to initialize
                
                n_batches = (len(X_train) + train_batch_size - 1) // train_batch_size
                for i in tqdm(range(n_batches), desc="Training Batches"):
                    start_idx = i * train_batch_size
                    end_idx = min((i + 1) * train_batch_size, len(X_train))
                    
                    X_batch = X_train.iloc[start_idx:end_idx]
                    y_batch = y_train.iloc[start_idx:end_idx]
                    
                    best_model.partial_fit(X_batch, y_batch, classes=classes)
                    
                    # Clear memory
                    del X_batch, y_batch
                    gc.collect()
            else:
                # For models that don't support partial_fit
                print("Training with full dataset (no batching)...")
                best_model.fit(X_train, y_train)
                
        except Exception as e:
            print(f"Error during training: {e}")
            print("Trying with reduced parameters...")
            try:
                # Try with even more conservative settings
                if hasattr(best_model, 'n_estimators') and best_model.n_estimators > 5:
                    print("Reducing number of estimators...")
                    best_model.n_estimators = 5
                if hasattr(best_model, 'max_depth') and best_model.max_depth is not None:
                    print("Reducing max depth...")
                    best_model.max_depth = min(5, best_model.max_depth)
                best_model.fit(X_train, y_train)
            except Exception as e2:
                print(f"Training failed: {e2}")
                print("Skipping this model...")
                return None, None
        
        # Evaluate on test set with proper class handling
        try:
            # Ensure we only predict classes that were seen during training
            y_pred = best_model.predict(self.X_test)
            
            # If any classes in test set weren't in training, default to majority class
            if hasattr(self, 'majority_class_'):
                y_pred = np.where(np.isin(y_pred, self.classes_), y_pred, self.majority_class_)
        except Exception as e:
            print(f"Error during prediction: {e}")
            print("Falling back to majority class prediction...")
            y_pred = np.full(len(self.X_test), self.majority_class_)
        
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
        
        # Save model with timestamp and metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_model_{timestamp}.pkl"
        model_path = self.model_dir / model_filename
        
        # Include additional metadata
        model_metadata = {
            'model': best_model,
            'metrics': metrics,
            'feature_names': list(self.X_train.columns) if hasattr(self.X_train, 'columns') else None,
            'timestamp': timestamp,
            'training_time': time.time() - self.training_start_time,
            'memory_usage': get_memory_usage(),
            'git_commit': self._get_git_commit(),
            'parameters': best_model.get_params()
        }
        
        # Save using joblib with compression
        joblib.dump(model_metadata, model_path, compress=('gzip', 3))
        print(f"\n✓ Model saved to {model_path} ({(model_path.stat().st_size / (1024**2)):.2f} MB)")
        
        # Save metrics as JSON
        metrics_path = self.model_dir / f"{model_name}_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        
        # Save feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            self._plot_feature_importance(best_model, model_name, timestamp)
            
        # Log resource usage
        self._log_resource_usage()
        
        # Plot confusion matrix
        self._plot_confusion_matrix(self.y_test, y_pred, model_name)
        
        return best_model, metrics
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """Plot and save confusion matrix with normalized values"""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
        ax1.set_title(f'Confusion Matrix - {model_name}\n(Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Plot normalized values
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax2)
        ax2.set_title('Normalized Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save figure
        cm_path = os.path.join(self.model_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to {cm_path}")
    
    def _plot_feature_importance(self, model, model_name: str, top_n: int = 20):
        """Plot and save feature importance"""
        if not hasattr(model, 'feature_importances_'):
            return
            
        # Get feature importances
        importances = model.feature_importances_
        
        # Get feature names
        if hasattr(self.X_train, 'columns'):
            feature_names = self.X_train.columns
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create a DataFrame for visualization
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
        plt.title(f'Top {top_n} Most Important Features - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save figure
        importance_path = os.path.join(self.model_dir, f"{model_name}_feature_importance.png")
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature importance plot saved to {importance_path}")
    
    def _plot_roc_curve(self, model, X_test, y_test, model_name: str):
        """Plot and save ROC curve"""
        if not hasattr(model, 'predict_proba'):
            return
            
        # Calculate ROC curve
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {model_name}')
        plt.legend(loc="lower right")
        
        # Save figure
        roc_path = os.path.join(self.model_dir, f"{model_name}_roc_curve.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curve saved to {roc_path}")
    
    def _plot_precision_recall_curve(self, model, X_test, y_test, model_name: str):
        """Plot and save Precision-Recall curve"""
        if not hasattr(model, 'predict_proba'):
            return
            
        # Calculate precision-recall curve
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.step(recall, precision, where='post', label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc='best')
        
        # Save figure
        pr_path = os.path.join(self.model_dir, f"{model_name}_precision_recall.png")
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Precision-Recall curve saved to {pr_path}")
    
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
    
    def _get_git_commit(self):
        """Get current git commit hash if available"""
        try:
            import subprocess
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except:
            return "unknown"

    def _plot_leaderboard(self, df_results: pd.DataFrame):
        """Plot model comparison"""
        import matplotlib.pyplot as plt
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
            
        for ax, metric in zip(axes, metrics):
            df_sorted = df_results.sort_values(metric, ascending=True)
            title = metric.replace('_', ' ').title()
            
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
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (default: 0.2)")
    parser.add_argument("--n-iter", type=int, default=20, help="Hyperparameter search iterations (default: 20)")
    parser.add_argument("--chunksize", type=int, default=50000, 
                       help="Number of rows to process at a time (default: 50000)")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU acceleration if available")
    
    # Model parameters
    parser.add_argument("--n-estimators", type=int, default=100,
                       help="Number of trees/estimators (default: 100)")
    parser.add_argument("--max-depth", type=int, default=6,
                       help="Maximum depth of the trees (default: 6)")
    
    # Random Forest specific parameters
    parser.add_argument("--max-features", type=float, default=0.5,
                       help="Fraction of features to consider (default: 0.5)")
    parser.add_argument("--min-samples-split", type=int, default=2,
                       help="Minimum samples required to split a node (default: 2)")
    parser.add_argument("--min-samples-leaf", type=int, default=1,
                       help="Minimum samples required at a leaf node (default: 1)")
    
    # XGBoost specific parameters
    parser.add_argument("--learning-rate", type=float, default=0.1,
                       help="Boosting learning rate (XGBoost, default: 0.1)")
    parser.add_argument("--min-child-weight", type=int, default=1,
                       help="Minimum sum of instance weight needed in a child (XGBoost, default: 1)")
    parser.add_argument("--gamma", type=float, default=0,
                       help="Minimum loss reduction required to make a further partition (XGBoost, default: 0)")
    parser.add_argument("--subsample", type=float, default=1.0,
                       help="Subsample ratio of the training instances (XGBoost, default: 1.0)")
    parser.add_argument("--colsample-bytree", type=float, default=1.0,
                       help="Subsample ratio of columns when constructing each tree (XGBoost, default: 1.0)")
    
    args = parser.parse_args()
    
    # Create trainer with GPU support if requested
    trainer = CICModelTrainer(
        data_path=args.data, 
        model_dir=args.model_dir,
        use_gpu=args.use_gpu
    )
    
    # Update model configurations with command line arguments
    trainer._initialize_model_configs(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=args.max_features,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree
    )
    
    # Load and split data with specified chunk size
    trainer.load_data(chunksize=args.chunksize)
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


    def _train_in_batches(self, model, batch_size=10000, checkpoint_interval=5):
        """Train model in batches to save memory"""
        # Determine number of batches
        n_samples = len(self.X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"Training in {n_batches} batches of {batch_size} samples")
        
        # Get model type for batch training
        model_type = type(model).__name__
        
        if model_type in ['RandomForestClassifier', 'GradientBoostingClassifier']:
            # These models support partial_fit or can handle large datasets
            for i in tqdm(range(0, n_samples, batch_size), desc="Training batches"):
                X_batch = self.X_train[i:i+batch_size]
                y_batch = self.y_train[i:i+batch_size]
                
                # For the first batch, initialize the model
                if i == 0:
                    model.fit(X_batch, y_batch)
                else:
                    # For subsequent batches, use warm start if available
                    if hasattr(model, 'warm_start') and hasattr(model, 'n_estimators'):
                        # Incrementally add more trees
                        model.n_estimators += 10
                        model.fit(X_batch, y_batch)
                    else:
                        # Otherwise, just refit on the current batch
                        model.fit(X_batch, y_batch)
                
                # Save checkpoint if needed
                if checkpoint_interval > 0 and (i // batch_size + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(model, f"{model_type}_batch_{i//batch_size}")
        else:
            # For models that don't support incremental learning, just fit on the whole dataset
            print(f"Model {model_type} doesn't support batch training. Fitting on full dataset.")
            model.fit(self.X_train, self.y_train)
        
        return model
    
    def _save_checkpoint(self, model, checkpoint_name):
        """Save model checkpoint"""
        checkpoint_path = self.model_dir / f"checkpoint_{checkpoint_name}.pkl"
        joblib.dump(model, checkpoint_path, compress=('gzip', 3))
        print(f"\n✓ Checkpoint saved to {checkpoint_path}")
    
    def _save_feature_importance(self, model, model_name, timestamp):
        """Save feature importance to CSV"""
        if not hasattr(model, 'feature_importances_'):
            return
            
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Get feature names
        if hasattr(self.X_train, 'columns'):
            feature_names = self.X_train.columns
        else:
            feature_names = [f'feature_{i}' for i in range(self.X_train.shape[1])]
        
        # Create and save importance DataFrame
        importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        })
        
        importance_path = self.model_dir / f"{model_name}_feature_importance_{timestamp}.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"✓ Feature importance saved to {importance_path}")
    
    def _get_git_commit(self):
        """Get current git commit hash"""
        try:
            import subprocess
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except:
            return "unknown"
    
    def _log_resource_usage(self):
        """Log system resource usage"""
        import psutil
        import platform
        
        mem = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'memory_used_mb': get_memory_usage(),
            'memory_percent': mem.percent,
            'cpu_percent': cpu_percent,
            'platform': platform.platform(),
            'python_version': platform.python_version()
        }
        
        log_path = self.model_dir / 'resource_usage.json'
        with open(log_path, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')


if __name__ == "__main__":
    main()
