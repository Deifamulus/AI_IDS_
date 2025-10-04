"""
CIC Dataset Preprocessing
This module preprocesses CIC-IDS datasets to be compatible with the existing training pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import os
from typing import Optional, List


class CICPreprocessor:
    """
    Preprocessor for CIC-IDS datasets
    Handles feature selection, encoding, scaling, and compatibility with existing pipeline
    """
    
    def __init__(self, artifacts_dir: str = "../artifacts"):
        """
        Initialize the preprocessor
        
        Args:
            artifacts_dir: Directory to save preprocessing artifacts (scalers, encoders)
        """
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)
        
        self.scaler = None
        self.label_encoder = None
        self.selected_features = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load preprocessed CIC data"""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_len = len(df)
        df = df.drop_duplicates()
        removed = initial_len - len(df)
        
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
        
        return df
    
    def handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values with large finite numbers"""
        print("Handling infinite values...")
        
        # Replace inf with NaN first
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # For each numeric column, replace NaN with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        return df
    
    def select_important_features(self, df: pd.DataFrame, 
                                 method: str = "variance",
                                 n_features: Optional[int] = None) -> pd.DataFrame:
        """
        Select important features from CIC dataset
        
        Args:
            df: Input DataFrame
            method: Feature selection method ('variance', 'correlation', 'all')
            n_features: Number of features to select (None = auto)
        
        Returns:
            DataFrame with selected features
        """
        print(f"\n--- Feature Selection (method: {method}) ---")
        
        # Separate features and labels
        label_col = 'label'
        if label_col not in df.columns:
            raise ValueError("Label column not found")
        
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        
        print(f"Starting with {len(X.columns)} numeric features")
        
        if method == "variance":
            # Remove low variance features
            variances = X.var()
            threshold = variances.quantile(0.1)  # Remove bottom 10%
            selected = variances[variances > threshold].index.tolist()
            
        elif method == "correlation":
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation > 0.95
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            selected = [col for col in X.columns if col not in to_drop]
            
        elif method == "all":
            selected = X.columns.tolist()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Limit number of features if specified
        if n_features and len(selected) > n_features:
            # Select top N by variance
            variances = X[selected].var().sort_values(ascending=False)
            selected = variances.head(n_features).index.tolist()
        
        self.selected_features = selected
        print(f"Selected {len(selected)} features")
        
        # Save selected features
        features_path = os.path.join(self.artifacts_dir, "selected_features.txt")
        with open(features_path, 'w') as f:
            f.write('\n'.join(selected))
        print(f"Selected features saved to {features_path}")
        
        # Return DataFrame with selected features + label
        return pd.concat([X[selected], y], axis=1)
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features (if any remain)
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with encoded features
        """
        # Identify categorical columns (excluding label)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if 'label' in cat_cols:
            cat_cols.remove('label')
        
        if cat_cols:
            print(f"\nEncoding {len(cat_cols)} categorical features: {cat_cols}")
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features using MinMaxScaler
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for test)
        
        Returns:
            DataFrame with scaled features
        """
        print("\n--- Scaling Features ---")
        
        # Separate features and labels
        X = df.drop(columns=['label'])
        y = df['label']
        
        # Get numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            self.scaler = MinMaxScaler()
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            
            # Save scaler
            scaler_path = os.path.join(self.artifacts_dir, "cic_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Set fit=True first.")
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        print(f"Scaled {len(numeric_cols)} numeric features")
        
        # Combine back with labels
        return pd.concat([X, y], axis=1)
    
    def encode_labels(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode labels to numeric values
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoder (True for training, False for test)
        
        Returns:
            DataFrame with encoded labels
        """
        print("\n--- Encoding Labels ---")
        
        if fit:
            self.label_encoder = LabelEncoder()
            df['label'] = self.label_encoder.fit_transform(df['label'])
            
            # Save encoder
            encoder_path = os.path.join(self.artifacts_dir, "cic_label_encoder.pkl")
            joblib.dump(self.label_encoder, encoder_path)
            print(f"Label encoder saved to {encoder_path}")
            
            # Print label mapping
            print("\nLabel mapping:")
            for i, label in enumerate(self.label_encoder.classes_):
                print(f"  {label} -> {i}")
        else:
            if self.label_encoder is None:
                raise ValueError("Label encoder not fitted. Set fit=True first.")
            df['label'] = self.label_encoder.transform(df['label'])
        
        return df
    
    def balance_dataset(self, df: pd.DataFrame, method: str = "undersample", 
                       max_samples_per_class: Optional[int] = None) -> pd.DataFrame:
        """
        Balance the dataset to handle class imbalance
        
        Args:
            df: Input DataFrame
            method: Balancing method ('undersample', 'none')
            max_samples_per_class: Maximum samples per class
        
        Returns:
            Balanced DataFrame
        """
        print("\n--- Balancing Dataset ---")
        
        label_counts = df['label'].value_counts()
        print("Class distribution before balancing:")
        print(label_counts)
        
        if method == "undersample":
            # Undersample majority classes
            if max_samples_per_class is None:
                max_samples_per_class = label_counts.min()
            
            balanced_dfs = []
            for label in df['label'].unique():
                label_df = df[df['label'] == label]
                if len(label_df) > max_samples_per_class:
                    label_df = label_df.sample(n=max_samples_per_class, random_state=42)
                balanced_dfs.append(label_df)
            
            df = pd.concat(balanced_dfs, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            
            print("\nClass distribution after balancing:")
            print(df['label'].value_counts())
        
        return df
    
    def preprocess_pipeline(self, input_path: str, output_path: str,
                          feature_selection: str = "variance",
                          n_features: Optional[int] = None,
                          balance: bool = False,
                          max_samples_per_class: Optional[int] = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for CIC dataset
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save processed data
            feature_selection: Feature selection method
            n_features: Number of features to select
            balance: Whether to balance the dataset
            max_samples_per_class: Max samples per class for balancing
        
        Returns:
            Processed DataFrame
        """
        print("=" * 60)
        print("CIC DATASET PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Load data
        df = self.load_data(input_path)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Handle infinite values
        df = self.handle_infinite_values(df)
        
        # Select features
        df = self.select_important_features(df, method=feature_selection, n_features=n_features)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Balance dataset (before scaling)
        if balance:
            df = self.balance_dataset(df, method="undersample", 
                                     max_samples_per_class=max_samples_per_class)
        
        # Scale features
        df = self.scale_features(df, fit=True)
        
        # Encode labels (do this last to keep readable labels during balancing)
        df = self.encode_labels(df, fit=True)
        
        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n{'=' * 60}")
        print(f"Processed data saved to: {output_path}")
        print(f"Final shape: {df.shape}")
        print(f"{'=' * 60}")
        
        return df


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess CIC-IDS dataset for training")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file (from cic_dataset_loader)")
    parser.add_argument("--output", type=str, default="../data/processed/cic_preprocessed.csv", 
                       help="Output file path")
    parser.add_argument("--artifacts-dir", type=str, default="../artifacts", 
                       help="Directory to save artifacts")
    parser.add_argument("--feature-selection", type=str, default="variance",
                       choices=["variance", "correlation", "all"],
                       help="Feature selection method")
    parser.add_argument("--n-features", type=int, help="Number of features to select")
    parser.add_argument("--balance", action="store_true", help="Balance the dataset")
    parser.add_argument("--max-samples", type=int, help="Max samples per class for balancing")
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = CICPreprocessor(artifacts_dir=args.artifacts_dir)
    
    # Run preprocessing pipeline
    preprocessor.preprocess_pipeline(
        input_path=args.input,
        output_path=args.output,
        feature_selection=args.feature_selection,
        n_features=args.n_features,
        balance=args.balance,
        max_samples_per_class=args.max_samples
    )
    
    print("\n✓ Preprocessing complete!")


if __name__ == "__main__":
    main()
