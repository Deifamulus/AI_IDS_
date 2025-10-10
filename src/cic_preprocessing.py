"""
CIC Dataset Preprocessing with Memory Optimization
This module preprocesses CIC-IDS datasets with memory efficiency
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import joblib
import os
import gc
from tqdm import tqdm
from typing import Optional, List, Dict, Union
import warnings
import pyarrow.parquet as pq
import json
warnings.filterwarnings('ignore')

class CICPreprocessor:
    def __init__(self, artifacts_dir: str = "artifacts"):
        """Initialize with memory-efficient settings"""
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.selected_features = None
        self.dtypes = self._get_optimized_dtypes()
        
    def _get_optimized_dtypes(self) -> Dict[str, str]:
        """Return optimized dtypes for memory efficiency"""
        return {
            'src_port': 'int32',
            'dst_port': 'int32',
            'protocol': 'int8',
            'flow_duration': 'int32',
            'tot_fwd_pkts': 'int32',
            'tot_bwd_pkts': 'int32',
            'totlen_fwd_pkts': 'float32',
            'totlen_bwd_pkts': 'float32',
            'fwd_pkt_len_max': 'float32',
            'fwd_pkt_len_min': 'float32',
            'fwd_pkt_len_mean': 'float32',
            'fwd_pkt_len_std': 'float32',
            'bwd_pkt_len_max': 'float32',
            'bwd_pkt_len_min': 'float32',
            'bwd_pkt_len_mean': 'float32',
            'bwd_pkt_len_std': 'float32',
            'flow_byts_s': 'float32',
            'flow_pkts_s': 'float32',
            'flow_iat_mean': 'float32',
            'flow_iat_std': 'float32',
            'flow_iat_max': 'float32',
            'flow_iat_min': 'float32',
            'fwd_iat_tot': 'float32',
            'fwd_iat_mean': 'float32',
            'fwd_iat_std': 'float32',
            'fwd_iat_max': 'float32',
            'fwd_iat_min': 'float32',
            'bwd_iat_tot': 'float32',
            'bwd_iat_mean': 'float32',
            'bwd_iat_std': 'float32',
            'bwd_iat_max': 'float32',
            'bwd_iat_min': 'float32',
            'fwd_psh_flags': 'int8',
            'bwd_psh_flags': 'int8',
            'fwd_urg_flags': 'int8',
            'bwd_urg_flags': 'int8',
            'fwd_header_len': 'int32',
            'bwd_header_len': 'int32',
            'fwd_pkts_s': 'float32',
            'bwd_pkts_s': 'float32',
            'pkt_len_min': 'float32',
            'pkt_len_max': 'float32',
            'pkt_len_mean': 'float32',
            'pkt_len_std': 'float32',
            'pkt_len_var': 'float32',
            'fin_flag_cnt': 'int8',
            'syn_flag_cnt': 'int8',
            'rst_flag_cnt': 'int8',
            'psh_flag_cnt': 'int8',
            'ack_flag_cnt': 'int8',
            'urg_flag_cnt': 'int8',
            'cwe_flag_count': 'int8',
            'ece_flag_cnt': 'int8',
            'down_up_ratio': 'float32',
            'pkt_size_avg': 'float32',
            'fwd_seg_size_avg': 'float32',
            'bwd_seg_size_avg': 'float32',
            'fwd_byts_b_avg': 'float32',
            'fwd_pkts_b_avg': 'float32',
            'fwd_blk_rate_avg': 'float32',
            'bwd_byts_b_avg': 'float32',
            'bwd_pkts_b_avg': 'float32',
            'bwd_blk_rate_avg': 'float32',
            'subflow_fwd_pkts': 'int32',
            'subflow_fwd_byts': 'int32',
            'subflow_bwd_pkts': 'int32',
            'subflow_bwd_byts': 'int32',
            'init_fwd_win_byts': 'int32',
            'init_bwd_win_byts': 'int32',
            'fwd_act_data_pkts': 'int32',
            'fwd_seg_size_min': 'float32',
            'active_mean': 'float32',
            'active_std': 'float32',
            'active_max': 'float32',
            'active_min': 'float32',
            'idle_mean': 'float32',
            'idle_std': 'float32',
            'idle_max': 'float32',
            'idle_min': 'float32',
            'label': 'category'
        }

    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        for col, dtype in self.dtypes.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    # Skip if conversion fails
                    continue
        return df

    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data"""
        # Handle missing values
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.fillna(chunk.median(numeric_only=True))
        return chunk

    def _read_file_in_chunks(self, filepath: str, columns: list = None, chunk_size: int = 100000):
        """Read file in chunks, handling both CSV and Parquet formats."""
        if filepath.endswith('.parquet'):
            # For parquet files
            parquet_file = pq.ParquetFile(filepath)
            for batch in parquet_file.iter_batches(columns=columns, batch_size=chunk_size):
                yield batch.to_pandas()
        else:
            # For CSV files
            for chunk in pd.read_csv(filepath, chunksize=chunk_size, usecols=columns, low_memory=False):
                yield chunk

    def preprocess_pipeline(self, input_path: str, output_path: str,
                          feature_selection: str = "variance",
                          n_features: Optional[int] = None,
                          balance: bool = False,
                          max_samples: int = 50000,
                          chunk_size: int = 100000) -> None:
        """Memory-efficient preprocessing pipeline with improved label handling"""
        print("Starting memory-efficient preprocessing...")
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # First, analyze the input data
        print("\nAnalyzing input data...")
        
        # Get all unique labels and their counts
        print("Counting unique labels in the dataset...")
        label_column = 'Label' if input_path.endswith('.parquet') else 'label'
        
        # Get label distribution
        label_counts = {}
        for chunk in self._read_file_in_chunks(input_path, columns=[label_column], chunk_size=chunk_size):
            chunk_counts = chunk[label_column].value_counts().to_dict()
            for k, v in chunk_counts.items():
                label_counts[k] = label_counts.get(k, 0) + v
        
        if not label_counts:
            raise ValueError("No data found in the input file")
            
        print("\nOriginal label distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{label}: {count:,} samples")
        
        # Process data in chunks
        print("\nProcessing data in chunks...")
        
        # First pass: Fit label encoder and feature selector if needed
        if not hasattr(self, 'label_encoder_fitted_'):
            print("\nFitting label encoder...")
            self.label_encoder.fit(list(label_counts.keys()))
            self.label_encoder_fitted_ = True
            
            # Save the label encoder
            os.makedirs(self.artifacts_dir, exist_ok=True)
            joblib.dump(
                self.label_encoder, 
                os.path.join(self.artifacts_dir, 'label_encoder.joblib')
            )
            print(f"Label encoder saved to {os.path.join(self.artifacts_dir, 'label_encoder.joblib')}")
        
        # Process data in chunks
        total_chunks = sum(1 for _ in self._read_file_in_chunks(input_path, chunk_size=chunk_size))
        first_chunk = True
        
        for chunk_idx, chunk in enumerate(tqdm(self._read_file_in_chunks(input_path, chunk_size=chunk_size), 
                                         total=total_chunks,
                                         desc="Processing data")):
            # Optimize memory usage
            chunk = self._optimize_memory(chunk)
            
            # Process the chunk
            processed_chunk = self._process_chunk(chunk)
            
            # Make a copy of the chunk to avoid SettingWithCopyWarning
            chunk_copy = processed_chunk.copy()
            
            # Transform labels
            try:
                y = self.label_encoder.transform(chunk_copy[label_column].astype(str).str.strip())
            except (KeyError, ValueError) as e:
                print(f"Error transforming labels: {e}")
                print("Refitting label encoder with new labels...")
                
                # Get all unique labels from the dataset including new ones
                all_labels = set()
                for label_chunk in self._read_file_in_chunks(input_path, columns=[label_column], chunk_size=100000):
                    all_labels.update(label_chunk[label_column].astype(str).str.strip().unique())
                
                # Add any existing classes from the label encoder
                if hasattr(self, 'label_encoder_fitted_'):
                    all_labels.update(self.label_encoder.classes_)
                
                # Reinitialize and fit the label encoder with all labels
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(list(all_labels))
                self.label_encoder_fitted_ = True
                
                # Save the updated encoder
                joblib.dump(
                    self.label_encoder,
                    os.path.join(self.artifacts_dir, 'label_encoder.joblib')
                )
                
                # Try transforming again
                y = self.label_encoder.transform(chunk_copy[label_column].astype(str).str.strip())
            
            # Drop label column before feature processing
            X = chunk_copy.drop(columns=[label_column])
            
            # Convert all columns to numeric, coercing errors to NaN
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # Fill any NaN values that resulted from conversion with 0
            X = X.fillna(0)
            
            # Feature selection
            if feature_selection == "variance" and n_features:
                if not hasattr(self, 'feature_selector_fitted_'):
                    print("\nPerforming feature selection...")
                    selector = VarianceThreshold(threshold=0.1)
                    X = selector.fit_transform(X)
                    self.feature_selector = selector
                    joblib.dump(
                        selector,
                        os.path.join(self.artifacts_dir, 'feature_selector.joblib')
                    )
                    self.feature_selector_fitted_ = True
                    print(f"Selected {X.shape[1]} features")
                else:
                    X = self.feature_selector.transform(X)
            
            # Scale features
            if not hasattr(self, 'scaler_fitted_'):
                print("\nFitting scaler...")
                X = self.scaler.fit_transform(X)
                joblib.dump(
                    self.scaler,
                    os.path.join(self.artifacts_dir, 'scaler.joblib')
                )
                self.scaler_fitted_ = True
            else:
                X = self.scaler.transform(X)
            
            # Store original feature names if this is the first chunk
            if first_chunk and not hasattr(self, 'feature_names_'):
                self.feature_names_ = chunk_copy.drop(columns=[label_column]).columns.tolist()
                
                # Save feature names to a JSON file
                feature_names_path = os.path.splitext(output_path)[0] + '_feature_names.json'
                with open(feature_names_path, 'w') as f:
                    json.dump({
                        'original_features': self.feature_names_,
                        'processed_features': [f'feature_{i}' for i in range(X.shape[1])]
                    }, f, indent=2)
                print(f"\nSaved feature names to {feature_names_path}")
            
            # Convert back to DataFrame and add labels
            processed_chunk = pd.DataFrame(
                X,
                columns=[f'feature_{i}' for i in range(X.shape[1])]
            )
            processed_chunk['label'] = y
            
            # Save chunk to disk
            processed_chunk.to_csv(
                output_path,
                mode='a',
                header=not os.path.exists(output_path) or first_chunk,
                index=False
            )
            first_chunk = False
            
            # Print progress
            if (chunk_idx + 1) % 10 == 0:
                print(f"Processed {(chunk_idx + 1) * chunk_size} samples...")
            
            del processed_chunk
            gc.collect()
        
        print(f"\nPreprocessing complete. Results saved to {output_path}")
        
        # Balance the dataset if requested
        if balance:
            print("\nBalancing dataset...")
            temp_path = f"{output_path}.temp"
            os.rename(output_path, temp_path)
            self.balance_dataset(temp_path, output_path, max_samples)
            os.remove(temp_path)
        
        # Print final label distribution
        print("\nFinal label distribution in preprocessed data:")
        final_label_counts = {}
        for chunk in pd.read_csv(output_path, chunksize=chunk_size, usecols=['label']):
            for label, count in chunk['label'].value_counts().items():
                final_label_counts[label] = final_label_counts.get(label, 0) + count
        
        for label, count in sorted(final_label_counts.items()):
            print(f"Class {label}: {count} samples")

    def balance_dataset(self, input_path: str, output_path: str, 
                       max_samples: int = 50000) -> None:
        """Balance dataset by undersampling majority classes"""
        print("Balancing dataset...")
        
        # Load the label encoder if it exists
        label_encoder_path = os.path.join(self.artifacts_dir, 'label_encoder.joblib')
        if os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)
        else:
            raise FileNotFoundError(f"Label encoder not found at {label_encoder_path}. Please run preprocessing first.")
        
        # Get all class names from the label encoder
        class_names = self.label_encoder.classes_
        print(f"Found {len(class_names)} classes in label encoder")
        
        # First pass: get class counts
        class_counts = {}
        for chunk in pd.read_csv(input_path, chunksize=100000, usecols=['label']):
            # Convert numeric labels back to original class names for counting
            chunk_labels = self.label_encoder.inverse_transform(chunk['label'].astype(int))
            for label, count in pd.Series(chunk_labels).value_counts().items():
                class_counts[label] = class_counts.get(label, 0) + count
        
        print(f"Original class distribution: {class_counts}")
        
        # Calculate samples per class (convert class names to encoded values)
        samples_per_class = {}
        for class_name, count in class_counts.items():
            encoded_label = self.label_encoder.transform([class_name])[0]
            samples_per_class[encoded_label] = min(count, max_samples)
        
        # Second pass: sample from each class
        first_chunk = True
        
        # Create a new CSV file (overwrite if exists)
        open(output_path, 'w').close()
        
        for encoded_label, n_samples in samples_per_class.items():
            # Get the original class name for logging
            class_name = self.label_encoder.inverse_transform([encoded_label])[0]
            print(f"\nProcessing class: {class_name} (encoded: {encoded_label}), target samples: {n_samples}")
            
            # Read and filter chunks for the current class
            label_chunks = []
            total_samples = 0
            
            for chunk in pd.read_csv(input_path, chunksize=100000, low_memory=False):
                # Filter rows where label matches the current encoded label
                chunk = chunk[chunk['label'].astype(int) == encoded_label].copy()
                
                if not chunk.empty:
                    # Calculate how many samples we can take from this chunk
                    remaining = n_samples - total_samples
                    if remaining <= 0:
                        break
                        
                    sample_size = min(remaining, len(chunk))
                    sampled_chunk = chunk.sample(n=sample_size, random_state=42)
                    label_chunks.append(sampled_chunk)
                    total_samples += len(sampled_chunk)
                    
                    if total_samples >= n_samples:
                        break
            
            if label_chunks:
                balanced_chunk = pd.concat(label_chunks)
                
                # Ensure the label column is properly encoded as integer
                balanced_chunk['label'] = balanced_chunk['label'].astype(int)
                
                # Save the chunk
                balanced_chunk.to_csv(
                    output_path,
                    mode='a',
                    header=first_chunk,
                    index=False
                )
                first_chunk = False
                
                print(f"Added {len(balanced_chunk)} samples for class {class_name} (encoded: {encoded_label})")
                
                # Clean up
                del balanced_chunk
                gc.collect()
        
        # Verify the balanced dataset
        print("\nVerifying balanced dataset...")
        balanced_counts = {}
        for chunk in pd.read_csv(output_path, chunksize=100000, usecols=['label']):
            # Convert numeric labels back to original class names for reporting
            chunk_labels = self.label_encoder.inverse_transform(chunk['label'].astype(int))
            for label, count in pd.Series(chunk_labels).value_counts().items():
                balanced_counts[label] = balanced_counts.get(label, 0) + count
        
        print("\nFinal class distribution:")
        for class_name, count in sorted(balanced_counts.items()):
            print(f"  {class_name}: {count} samples")
            
        print(f"\nTotal samples in balanced dataset: {sum(balanced_counts.values())}")
        print(f"Balanced dataset saved to {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Memory-efficient CIC dataset preprocessing')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--feature-selection', type=str, default='variance',
                       choices=['variance', 'all'], help='Feature selection method')
    parser.add_argument('--n-features', type=int, default=None, 
                       help='Number of features to select')
    parser.add_argument('--balance', action='store_true', help='Balance the dataset')
    parser.add_argument('--max-samples', type=int, default=50000,
                       help='Maximum samples per class for balancing')
    parser.add_argument('--chunk-size', type=int, default=100000,
                       help='Number of rows to process at a time')
    
    args = parser.parse_args()
    
    preprocessor = CICPreprocessor()
    
    # Preprocess data
    temp_path = f"{args.output}.temp"
    preprocessor.preprocess_pipeline(
        input_path=args.input,
        output_path=temp_path,
        feature_selection=args.feature_selection,
        n_features=args.n_features,
        chunk_size=args.chunk_size
    )
    
    # Balance if requested
    if args.balance:
        preprocessor.balance_dataset(
            input_path=temp_path,
            output_path=args.output,
            max_samples=args.max_samples
        )
        os.remove(temp_path)  # Clean up temporary file
    else:
        os.rename(temp_path, args.output)

if __name__ == "__main__":
    main()

    