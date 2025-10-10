"""
Script to check the class distribution in a dataset.
"""
import pandas as pd
import argparse
from tqdm import tqdm

def check_distribution(file_path: str, label_col: str = 'label', chunksize: int = 100000) -> dict:
    """
    Check the distribution of labels in a CSV file.
    
    Args:
        file_path: Path to the CSV file
        label_col: Name of the label column
        chunksize: Number of rows to read at a time
        
    Returns:
        Dictionary with label counts
    """
    print(f"Checking distribution in {file_path}...")
    
    # Get the total number of rows for the progress bar
    total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='ignore')) - 1  # Subtract header
    
    label_counts = {}
    
    # Read the file in chunks
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
            # Update label counts
            for label, count in chunk[label_col].value_counts().items():
                label_counts[label] = label_counts.get(label, 0) + count
            
            # Update progress bar
            pbar.update(len(chunk))
    
    return label_counts

def main():
    parser = argparse.ArgumentParser(description='Check class distribution in a dataset')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--label-col', type=str, default='label', help='Name of the label column')
    parser.add_argument('--chunksize', type=int, default=100000, help='Number of rows to process at a time')
    
    args = parser.parse_args()
    
    # Check distribution
    distribution = check_distribution(
        file_path=args.input,
        label_col=args.label_col,
        chunksize=args.chunksize
    )
    
    # Print results
    print("\nClass distribution:")
    print("-" * 50)
    print(f"{'Class':<20} {'Count':<15} {'Percentage':<15}")
    print("-" * 50)
    
    total = sum(distribution.values())
    for label in sorted(distribution.keys()):
        count = distribution[label]
        percentage = (count / total) * 100
        print(f"{str(label):<20} {count:<15,} {percentage:.2f}%")
    
    print("-" * 50)
    print(f"{'Total':<20} {total:,}")
    print("-" * 50)

if __name__ == "__main__":
    main()
