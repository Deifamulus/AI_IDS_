import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict
import os
import gc

def analyze_distribution(filepath, chunk_size=100000):
    print(f"Analyzing {filepath}...")
    
    # Get total rows
    pf = pq.ParquetFile(filepath)
    total_rows = pf.metadata.num_rows
    print(f"Total rows: {total_rows:,}")
    
    # Initialize counters
    label_counts = defaultdict(int)
    total_chunks = (total_rows // chunk_size) + 1
    
    # Analyze in chunks
    print("\nReading data in batches...")
    for i, batch in enumerate(pf.iter_batches(batch_size=chunk_size, columns=None)):
        try:
            df = batch.to_pandas()
            
            # Try to identify label column (case insensitive)
            label_col = next((col for col in df.columns if col.lower() in ['label', 'attack', 'class', 'attacktype', 'attack_type']), None)
            if label_col is None:
                print("\nWarning: Could not find standard label column. Using first column as label.")
                label_col = df.columns[0]
                print(f"Using column '{label_col}' as label. First few values: {df[label_col].head(3).tolist()}")
                
            # Update counts
            for label, count in df[label_col].value_counts().items():
                label_counts[str(label).strip()] += count
                
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {(i + 1) * chunk_size:,} rows...")
                
            # Clean up memory
            del df
            gc.collect()
            
        except Exception as e:
            print(f"\nError processing batch {i}: {str(e)}")
            print(f"Batch columns: {batch.schema.names}")
            continue
    
    # Print distribution
    print("\nClass distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {count:,} samples")
    
    # Calculate statistics
    min_samples = min(label_counts.values())
    max_samples = max(label_counts.values())
    avg_samples = sum(label_counts.values()) / len(label_counts)
    
    print(f"\nStatistics:")
    print(f"Number of classes: {len(label_counts)}")
    print(f"Minimum samples per class: {min_samples:,}")
    print(f"Maximum samples per class: {max_samples:,}")
    print(f"Average samples per class: {avg_samples:,.1f}")
    
    # Calculate safe sample size (80% of the smallest class, but cap at 50,000 for practicality)
    safe_sample_size = min(int(min_samples * 0.8), 50000)
    print(f"\nRecommended max_samples for balancing: {safe_sample_size:,}")
    
    # Show what this would give us
    balanced_size = safe_sample_size * len(label_counts)
    print(f"This would create a balanced dataset of ~{balanced_size:,} total samples")
    
    # Alternative approach: weighted sampling
    print("\nAlternative balancing strategy (weighted sampling):")
    print("1. Keep all samples from rare classes (n < 1,000)")
    print("2. Cap majority classes at 10,000 samples")
    print("3. Use class weights during training")
    
    # Calculate samples for this strategy
    total_samples = 0
    for label, count in label_counts.items():
        if count < 1000:
            total_samples += count  # Keep all samples
        else:
            total_samples += min(count, 10000)  # Cap at 10,000
    
    print(f"\nThis would create a dataset of ~{total_samples:,} total samples")
    print("with better representation of rare classes.")
    
    return label_counts

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_distribution.py <path_to_parquet>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
        
    analyze_distribution(filepath)
