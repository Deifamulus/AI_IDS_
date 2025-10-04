"""
Complete CIC-IDS Training Pipeline
This script runs the entire pipeline from raw CIC data to trained models
"""

import os
import sys
import argparse
from datetime import datetime

# Import our custom modules
from cic_dataset_loader import CICDatasetLoader
from cic_preprocessing import CICPreprocessor
from train_cic_model import CICModelTrainer


def run_pipeline(args):
    """
    Run the complete CIC-IDS training pipeline
    
    Args:
        args: Command line arguments
    """
    print("=" * 70)
    print("CIC-IDS COMPLETE TRAINING PIPELINE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create output directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Define intermediate file paths
    cleaned_data_path = os.path.join("data", "cic_cleaned.csv")
    preprocessed_data_path = os.path.join("data", "processed", "cic_preprocessed.csv")
    
    # ========================================================================
    # STEP 1: Load and Clean CIC Dataset
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: LOADING AND CLEANING CIC DATASET")
    print("=" * 70)
    
    loader = CICDatasetLoader(args.cic_path, args.dataset_type)
    
    try:
        # Load dataset
        loader.load_dataset()
        
        # Clean dataset
        loader.clean_dataset()
        
        # Show label distribution
        loader.get_label_distribution()
        
        # Standardize labels
        loader.standardize_labels(binary=args.binary)
        
        # Sample if requested
        if args.sample_frac or args.sample_n:
            print(f"\nSampling dataset...")
            loader.sample_dataset(
                n_samples=args.sample_n, 
                frac=args.sample_frac, 
                stratify=True
            )
        
        # Save cleaned data
        loader.save_processed_data(cleaned_data_path)
        
        print(f"\n✓ Step 1 Complete: Data cleaned and saved to {cleaned_data_path}")
        
    except Exception as e:
        print(f"\n✗ Error in Step 1: {e}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: Preprocess Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING DATA")
    print("=" * 70)
    
    preprocessor = CICPreprocessor(artifacts_dir=args.artifacts_dir)
    
    try:
        preprocessor.preprocess_pipeline(
            input_path=cleaned_data_path,
            output_path=preprocessed_data_path,
            feature_selection=args.feature_selection,
            n_features=args.n_features,
            balance=args.balance,
            max_samples_per_class=args.max_samples
        )
        
        print(f"\n✓ Step 2 Complete: Data preprocessed and saved to {preprocessed_data_path}")
        
    except Exception as e:
        print(f"\n✗ Error in Step 2: {e}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 3: Train Models
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING MODELS")
    print("=" * 70)
    
    trainer = CICModelTrainer(
        data_path=preprocessed_data_path,
        model_dir=args.model_dir
    )
    
    try:
        # Load and split data
        trainer.load_data()
        trainer.split_data(test_size=args.test_size)
        
        # Train models
        if args.model == "all":
            results = trainer.train_all_models(
                use_smote=not args.no_smote,
                tune_hyperparams=not args.no_tune
            )
        else:
            model, metrics = trainer.train_model(
                args.model,
                use_smote=not args.no_smote,
                tune_hyperparams=not args.no_tune,
                n_iter=args.n_iter
            )
            results = [metrics]
        
        print(f"\n✓ Step 3 Complete: Models trained and saved to {args.model_dir}")
        
    except Exception as e:
        print(f"\n✗ Error in Step 3: {e}")
        sys.exit(1)
    
    # ========================================================================
    # Pipeline Complete
    # ========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Output Summary:")
    print(f"  - Cleaned data:      {cleaned_data_path}")
    print(f"  - Preprocessed data: {preprocessed_data_path}")
    print(f"  - Artifacts:         {args.artifacts_dir}/")
    print(f"  - Models:            {args.model_dir}/")
    print()
    print("Next Steps:")
    print("  1. Check the leaderboard: models/cic_models/leaderboard.csv")
    print("  2. View confusion matrices: models/cic_models/*_confusion_matrix.png")
    print("  3. Use trained models for prediction")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Complete CIC-IDS training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with binary classification
  python train_cic_pipeline.py --cic-path data/cic_raw/ --binary

  # With sampling for quick testing
  python train_cic_pipeline.py --cic-path data/cic_raw/ --sample-frac 0.1 --binary

  # Multi-class with balanced dataset
  python train_cic_pipeline.py --cic-path data/cic_raw/ --balance --max-samples 50000

  # Train specific model without tuning (fast)
  python train_cic_pipeline.py --cic-path data/cic_raw/ --model random_forest --no-tune
        """
    )
    
    # Dataset arguments
    parser.add_argument("--cic-path", type=str, required=True,
                       help="Path to CIC dataset (file or directory)")
    parser.add_argument("--dataset-type", type=str, default="CIC-IDS2017",
                       help="Dataset type (CIC-IDS2017, CIC-IDS2018, etc.)")
    
    # Data loading arguments
    parser.add_argument("--binary", action="store_true",
                       help="Convert to binary classification (BENIGN vs ATTACK)")
    parser.add_argument("--sample-frac", type=float,
                       help="Sample fraction of dataset (0.0-1.0)")
    parser.add_argument("--sample-n", type=int,
                       help="Sample N rows from dataset")
    
    # Preprocessing arguments
    parser.add_argument("--feature-selection", type=str, default="variance",
                       choices=["variance", "correlation", "all"],
                       help="Feature selection method")
    parser.add_argument("--n-features", type=int,
                       help="Number of features to select")
    parser.add_argument("--balance", action="store_true",
                       help="Balance the dataset using undersampling")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum samples per class for balancing")
    
    # Training arguments
    parser.add_argument("--model", type=str, default="all",
                       choices=["all", "random_forest", "xgboost", "logistic_regression",
                               "gradient_boosting", "knn", "decision_tree"],
                       help="Model to train")
    parser.add_argument("--no-smote", action="store_true",
                       help="Disable SMOTE oversampling")
    parser.add_argument("--no-tune", action="store_true",
                       help="Disable hyperparameter tuning")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size (default: 0.2)")
    parser.add_argument("--n-iter", type=int, default=20,
                       help="Hyperparameter search iterations (default: 20)")
    
    # Output arguments
    parser.add_argument("--artifacts-dir", type=str, default="artifacts",
                       help="Directory to save preprocessing artifacts")
    parser.add_argument("--model-dir", type=str, default="models/cic_models",
                       help="Directory to save trained models")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.sample_frac and (args.sample_frac <= 0 or args.sample_frac > 1):
        parser.error("--sample-frac must be between 0 and 1")
    
    if args.test_size <= 0 or args.test_size >= 1:
        parser.error("--test-size must be between 0 and 1")
    
    # Run pipeline
    run_pipeline(args)


if __name__ == "__main__":
    main()
