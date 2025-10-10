"""
Analyze CIC dataset class distribution and suggest optimal max_samples value.
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

def analyze_distribution(filepath):
    print(f"Analyzing {filepath}...")
    
    # For parquet files
    if filepath.endswith('.parquet'):
        try:
            # Read parquet metadata to get schema and row count
            parquet_file = pq.ParquetFile(filepath)
            total_rows = parquet_file.metadata.num_rows
            print(f"Total rows: {total_rows:,}")
            
            # Get all column names
            column_names = parquet_file.schema.names
            print(f"\nAvailable columns: {', '.join(column_names)}")
            
            # Try to find the label column (case insensitive)
            label_columns = [col for col in column_names if 'label' in col.lower() or 'attack' in col.lower() or 'class' in col.lower()]
            
            if not label_columns:
                print("\n❌ Could not find a suitable label column. Please specify which column to use as the label.")
                print("Available columns:", ", ".join(column_names))
                return
                
            # Use the first matching column as label (or ask user to choose if multiple)
            label_column = label_columns[0]
            if len(label_columns) > 1:
                print(f"\nFound multiple potential label columns: {', '.join(label_columns)}")
                print(f"Using column '{label_column}' as the label. If this is incorrect, please specify the correct column name.")
            
            print(f"\nAnalyzing distribution of column: {label_column}")
            
            # Read just the label column to save memory
            label_counts = {}
            
            # Process in batches
            batch_size = 100000
            num_batches = (total_rows // batch_size) + 1
            
            print("\nReading data in batches...")
            for batch in tqdm(parquet_file.iter_batches(batch_size=batch_size, columns=[label_column]), 
                            total=num_batches, 
                            unit='batch'):
                # Convert batch to pandas Series
                labels = batch.column(label_column).to_pandas()
                # Update counts
                for label, count in labels.value_counts().items():
                    label = str(label).strip()  # Convert to string and clean whitespace
                    label_counts[label] = label_counts.get(label, 0) + count
    
        except Exception as e:
            print(f"Error reading parquet file: {e}")
            return
    
    # For CSV files
    elif filepath.endswith('.csv'):
        try:
            # Get total rows (approximate for CSV)
            with open(filepath, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1  # Subtract header
            print(f"Total rows: {total_rows:,}")
            
            # Read in chunks
            label_counts = {}
            chunk_size = 100000
            
            print("Reading data in chunks...")
            for chunk in tqdm(pd.read_csv(filepath, usecols=['label'], chunksize=chunk_size),
                            total=(total_rows//chunk_size) + 1,
                            unit='chunk'):
                # Update counts
                for label, count in chunk['label'].value_counts().items():
                    label_counts[label] = label_counts.get(label, 0) + count
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return
    else:
        print("Unsupported file format. Please provide a .parquet or .csv file.")
        return
    
    # Calculate statistics
    if not label_counts:
        print("No data found in the file")
        return
    
    # Convert to DataFrame for better display
    if not label_counts:
        print("\n❌ No data found in the specified label column.")
        return
        
    df = pd.DataFrame({
        'class': [str(k) for k in label_counts.keys()],  # Ensure all keys are strings
        'samples': label_counts.values()
    }).sort_values('samples', ascending=False)
    
    # Calculate percentages
    df['percentage'] = (df['samples'] / df['samples'].sum() * 100).round(2)
    
    print("\nClass Distribution:")
    print("-" * 60)
    print(f"{'Class':<30} {'Samples':>12} {'% of Total':>12} {'Cumulative %':>12}")
    print("-" * 60)
    
    cum_percent = 0
    for _, row in df.iterrows():
        cum_percent += row['percentage']
        print(f"{str(row['class']):<30} {row['samples']:>12,} {row['percentage']:>11.2f}% {cum_percent:>11.2f}%")
    
    print("-" * 60)
    print(f"{'Total':<30} {df['samples'].sum():>12,} {'100.00%':>11} {'100.00%':>12}")
    
    # Calculate statistics
    min_samples = df['samples'].min()
    max_samples = df['samples'].max()
    median_samples = df['samples'].median()
    mean_samples = df['samples'].mean()
    
    print("\nStatistics:")
    print("-" * 50)
    print(f"{'Number of classes:':<30} {len(df):>15,}")
    print(f"{'Total samples:':<30} {df['samples'].sum():>15,}")
    print(f"{'Minimum samples/class:':<30} {min_samples:>15,}")
    print(f"{'Maximum samples/class:':<30} {max_samples:>15,}")
    print(f"{'Median samples/class:':<30} {median_samples:>15.0f}")
    print(f"{'Mean samples/class:':<30} {mean_samples:>15.0f}")
    
    # Calculate class imbalance ratio
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    print(f"{'Class imbalance ratio:':<30} {imbalance_ratio:>15.1f}x")
    
    # Suggest max_samples based on distribution
    if len(df) > 10:  # Multi-class with many classes
        # Use the 60th percentile or 1.5x median, whichever is smaller
        q60 = df['samples'].quantile(0.6)
        suggested_max = min(int(median_samples * 1.5), int(q60))
    else:  # Binary or few classes
        # Use the median or 50,000, whichever is smaller
        suggested_max = min(50000, int(median_samples))
    
    # Adjust based on imbalance ratio
    if imbalance_ratio > 100:  # Highly imbalanced
        # Be more conservative with majority classes
        suggested_max = min(suggested_max, int(median_samples * 1.2))
    
    # Ensure reasonable bounds
    suggested_max = max(1000, min(suggested_max, 50000))
    
    print("\nRecommendations:")
    print("-" * 50)
    print(f"Suggested --max-samples: {suggested_max:,}")
    print("\nThis value is chosen to:")
    print(f"- Balance class distribution")
    print(f"- Maintain sufficient samples per class")
    print(f"- Prevent any single class from dominating the training")
    
    # Additional suggestions based on distribution
    if imbalance_ratio > 50:
        print("\n⚠️  High class imbalance detected! Consider:")
        print("- Using class weights in your model")
        print("- Trying oversampling techniques for minority classes")
        print("- Using F1 score as your primary metric instead of accuracy")
    
    print("\nExample usage:")
    print(f"python src/cic_preprocessing.py --input {filepath} --output output.csv")
    print(f"  --feature-selection variance --n-features 50 --balance --max-samples {suggested_max}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_cic.py <path_to_dataset>")
        print("Example: python analyze_cic.py data/raw/cic-collection.parquet")
        sys.exit(1)
    
    analyze_distribution(sys.argv[1])
