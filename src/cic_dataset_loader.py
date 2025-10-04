"""
CIC-IDS Dataset Loader
This module handles loading and initial processing of CIC-IDS datasets (2017, 2018, etc.)
"""

import pandas as pd
import os
import glob
from typing import List, Dict, Optional
import numpy as np


class CICDatasetLoader:
    """
    Loader for CIC-IDS datasets (CIC-IDS2017, CIC-IDS2018, CSE-CIC-IDS2018, etc.)
    """
    
    def __init__(self, dataset_path: str, dataset_type: str = "CIC-IDS2017"):
        """
        Initialize the CIC dataset loader
        
        Args:
            dataset_path: Path to the CIC dataset directory or CSV file
            dataset_type: Type of CIC dataset (CIC-IDS2017, CIC-IDS2018, etc.)
        """
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.df = None
        
    def load_single_csv(self, file_path: str) -> pd.DataFrame:
        """Load a single CSV file from CIC dataset"""
        print(f"Loading {file_path}...")
        
        try:
            # CIC datasets often have encoding issues
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
        
        # Clean column names (remove spaces, special characters)
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        
        return df
    
    def load_parquet(self, file_path: str) -> pd.DataFrame:
        """Load a Parquet file from CIC dataset"""
        print(f"Loading Parquet file: {file_path}...")
        
        try:
            df = pd.read_parquet(file_path)
            print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            
            # Clean column names (remove spaces, special characters)
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
            
            return df
        except Exception as e:
            print(f"Error loading Parquet file {file_path}: {e}")
            return pd.DataFrame()
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Load the entire CIC dataset (single file or multiple files)
        Supports CSV and Parquet formats
        
        Returns:
            Combined DataFrame with all data
        """
        if os.path.isfile(self.dataset_path):
            # Single file - check extension
            if self.dataset_path.endswith('.parquet'):
                self.df = self.load_parquet(self.dataset_path)
            else:
                self.df = self.load_single_csv(self.dataset_path)
        elif os.path.isdir(self.dataset_path):
            # Directory with multiple files
            csv_files = glob.glob(os.path.join(self.dataset_path, "*.csv"))
            parquet_files = glob.glob(os.path.join(self.dataset_path, "*.parquet"))
            
            if not csv_files and not parquet_files:
                raise ValueError(f"No CSV or Parquet files found in {self.dataset_path}")
            
            print(f"Found {len(csv_files)} CSV files and {len(parquet_files)} Parquet files")
            dfs = []
            
            # Load CSV files
            for csv_file in csv_files:
                df = self.load_single_csv(csv_file)
                if not df.empty:
                    dfs.append(df)
            
            # Load Parquet files
            for parquet_file in parquet_files:
                df = self.load_parquet(parquet_file)
                if not df.empty:
                    dfs.append(df)
            
            if dfs:
                self.df = pd.concat(dfs, ignore_index=True)
            else:
                raise ValueError("No valid data loaded from files")
        else:
            raise ValueError(f"Invalid path: {self.dataset_path}")
        
        print(f"\nDataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {len(self.df.columns)}")
        
        return self.df
    
    def clean_dataset(self) -> pd.DataFrame:
        """
        Clean the CIC dataset (handle missing values, infinities, etc.)
        
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print("\n--- Cleaning Dataset ---")
        initial_shape = self.df.shape
        
        # Replace infinity values with NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Check for missing values
        missing_before = self.df.isnull().sum().sum()
        print(f"Missing values before cleaning: {missing_before}")
        
        # Drop rows with missing values in critical columns
        # For CIC datasets, the label column is usually named 'label' or 'Label'
        label_col = self._find_label_column()
        
        if label_col:
            # Drop rows where label is missing
            self.df.dropna(subset=[label_col], inplace=True)
        
        # Fill remaining missing values with 0 (common for network features)
        self.df.fillna(0, inplace=True)
        
        missing_after = self.df.isnull().sum().sum()
        print(f"Missing values after cleaning: {missing_after}")
        print(f"Shape after cleaning: {self.df.shape} (removed {initial_shape[0] - self.df.shape[0]} rows)")
        
        return self.df
    
    def _find_label_column(self) -> Optional[str]:
        """Find the label column in the dataset"""
        possible_names = ['label', 'Label', 'class', 'Class', 'attack', 'Attack']
        
        for col in self.df.columns:
            if col in possible_names:
                return col
        
        # Check if any column contains 'label' or 'class'
        for col in self.df.columns:
            if 'label' in col.lower() or 'class' in col.lower():
                return col
        
        return None
    
    def get_label_distribution(self) -> pd.Series:
        """Get the distribution of attack types/labels"""
        label_col = self._find_label_column()
        
        if label_col:
            print(f"\n--- Label Distribution (Column: {label_col}) ---")
            dist = self.df[label_col].value_counts()
            print(dist)
            print(f"\nTotal samples: {len(self.df)}")
            print(f"Number of classes: {len(dist)}")
            return dist
        else:
            print("Warning: Label column not found")
            return pd.Series()
    
    def standardize_labels(self, binary: bool = False) -> pd.DataFrame:
        """
        Standardize label names and optionally convert to binary classification
        
        Args:
            binary: If True, convert to binary (normal vs attack)
        
        Returns:
            DataFrame with standardized labels
        """
        label_col = self._find_label_column()
        
        if not label_col:
            raise ValueError("Label column not found")
        
        # Clean label values
        self.df[label_col] = self.df[label_col].str.strip().str.upper()
        
        if binary:
            print("\nConverting to binary classification (BENIGN vs ATTACK)")
            # Map all non-benign labels to 'ATTACK'
            self.df[label_col] = self.df[label_col].apply(
                lambda x: 'BENIGN' if x in ['BENIGN', 'NORMAL'] else 'ATTACK'
            )
        
        # Rename column to 'label' for consistency
        if label_col != 'label':
            self.df.rename(columns={label_col: 'label'}, inplace=True)
        
        print("\nLabel distribution after standardization:")
        print(self.df['label'].value_counts())
        
        return self.df
    
    def sample_dataset(self, n_samples: Optional[int] = None, 
                      frac: Optional[float] = None, 
                      stratify: bool = True) -> pd.DataFrame:
        """
        Sample the dataset for faster experimentation
        
        Args:
            n_samples: Number of samples to take
            frac: Fraction of dataset to sample (0.0 to 1.0)
            stratify: Whether to maintain class distribution
        
        Returns:
            Sampled DataFrame
        """
        if self.df is None:
            raise ValueError("Dataset not loaded")
        
        if n_samples is None and frac is None:
            raise ValueError("Either n_samples or frac must be specified")
        
        label_col = 'label' if 'label' in self.df.columns else self._find_label_column()
        
        if stratify and label_col:
            if n_samples:
                self.df = self.df.groupby(label_col, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), n_samples // self.df[label_col].nunique()))
                )
            elif frac:
                self.df = self.df.groupby(label_col, group_keys=False).apply(
                    lambda x: x.sample(frac=frac)
                )
        else:
            if n_samples:
                self.df = self.df.sample(n=min(n_samples, len(self.df)))
            elif frac:
                self.df = self.df.sample(frac=frac)
        
        print(f"\nSampled dataset shape: {self.df.shape}")
        return self.df
    
    def save_processed_data(self, output_path: str):
        """Save the processed dataset"""
        if self.df is None:
            raise ValueError("No data to save")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"\nProcessed data saved to: {output_path}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and process CIC-IDS dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to CIC dataset (file or directory)")
    parser.add_argument("--output", type=str, default="../data/cic_processed.csv", help="Output file path")
    parser.add_argument("--dataset-type", type=str, default="CIC-IDS2017", help="Dataset type")
    parser.add_argument("--binary", action="store_true", help="Convert to binary classification")
    parser.add_argument("--sample-frac", type=float, help="Sample fraction of dataset (0.0-1.0)")
    parser.add_argument("--sample-n", type=int, help="Sample N rows from dataset")
    
    args = parser.parse_args()
    
    # Load dataset
    loader = CICDatasetLoader(args.input, args.dataset_type)
    loader.load_dataset()
    
    # Clean dataset
    loader.clean_dataset()
    
    # Show label distribution
    loader.get_label_distribution()
    
    # Standardize labels
    loader.standardize_labels(binary=args.binary)
    
    # Sample if requested
    if args.sample_frac or args.sample_n:
        loader.sample_dataset(n_samples=args.sample_n, frac=args.sample_frac, stratify=True)
    
    # Save processed data
    loader.save_processed_data(args.output)
    
    print("\n✓ Dataset loading complete!")


if __name__ == "__main__":
    main()
