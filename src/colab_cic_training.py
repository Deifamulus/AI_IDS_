"""
Colab-Friendly Single Script: Load CIC dataset (CSV/Parquet), preprocess, train, and save models/artifacts

This script wraps the existing modules:
- cic_dataset_loader.CICDatasetLoader
- cic_preprocessing.CICPreprocessor
- train_cic_model.CICModelTrainer

Usage in Google Colab (example):

!python src/colab_cic_training.py \
  --input "/content/drive/MyDrive/datasets/cic/cic-collection.parquet" \
  --output-dir "/content/drive/MyDrive/ai_ids_outputs" \
  --binary \
  --sample-frac 0.1 \
  --feature-selection variance \
  --n-features 50 \
  --balance \
  --max-samples 50000 \
  --model all \
  --n-iter 20

Notes:
- Add --mount-drive to mount your Google Drive automatically.
- If running first time on Colab, add --install-req to pip install requirements.
"""

import os
import sys
import argparse
from datetime import datetime

# Colab helpers

def in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def maybe_mount_drive(auto_mount: bool):
    if auto_mount and in_colab():
        from google.colab import drive  # type: ignore
        print("[i] Mounting Google Drive at /content/drive ...")
        drive.mount('/content/drive')
        print("[✓] Drive mounted")


def maybe_install_requirements(install_req: bool, project_root: str):
    if install_req and in_colab():
        req_path = os.path.join(project_root, 'requirements.txt')
        if os.path.exists(req_path):
            print(f"[i] Installing requirements from {req_path} ...")
            exit_code = os.system(f"pip install -q -r \"{req_path}\"")
            if exit_code != 0:
                print("[!] pip install returned non-zero exit code. You may need to install packages manually.")
        else:
            print(f"[!] requirements.txt not found at {req_path}. Skipping auto install.")


# Ensure we can import local modules regardless of working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import project modules
from cic_dataset_loader import CICDatasetLoader
from cic_preprocessing import CICPreprocessor
from train_cic_model import CICModelTrainer


def run_pipeline(
    input_path: str,
    output_dir: str,
    dataset_type: str,
    binary: bool,
    sample_frac: float | None,
    feature_selection: str,
    n_features: int | None,
    balance: bool,
    max_samples: int | None,
    model: str,
    no_smote: bool,
    no_tune: bool,
    test_size: float,
    n_iter: int,
    min_samples: int,
    mount_drive: bool,
    install_req: bool,
):
    # Optionally mount drive and install requirements
    maybe_mount_drive(mount_drive)
    maybe_install_requirements(install_req, PROJECT_ROOT)

    # Prepare output folders
    artifacts_dir = os.path.join(output_dir, 'artifacts')
    model_dir = os.path.join(output_dir, 'models', 'cic_models')
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    cleaned_data_path = os.path.join(data_dir, 'cic_cleaned.csv')
    preprocessed_data_path = os.path.join(data_dir, 'cic_preprocessed.csv')

    print("=" * 80)
    print("CIC-IDS COLAB TRAINING SCRIPT")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input path: {input_path}")
    print(f"Output dir: {output_dir}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Step 1: Load + Clean + Standardize
    print("\n" + "-" * 80)
    print("STEP 1: LOADING AND CLEANING DATA")
    print("-" * 80)
    loader = CICDatasetLoader(input_path, dataset_type)
    df = loader.load_dataset()
    df = loader.clean_dataset()
    loader.get_label_distribution()
    df = loader.standardize_labels(binary=binary)

    if sample_frac:
        print(f"[i] Applying stratified sampling (frac={sample_frac}) ...")
        df = loader.sample_dataset(frac=sample_frac, stratify=True)

    loader.save_processed_data(cleaned_data_path)

    # Step 2: Preprocess
    print("\n" + "-" * 80)
    print("STEP 2: PREPROCESSING DATA")
    print("-" * 80)
    preprocessor = CICPreprocessor(artifacts_dir=artifacts_dir)
    df_processed = preprocessor.preprocess_pipeline(
        input_path=cleaned_data_path,
        output_path=preprocessed_data_path,
        feature_selection=feature_selection,
        n_features=n_features,
        balance=balance,
        max_samples_per_class=max_samples,
    )

    # Step 3: Train
    print("\n" + "-" * 80)
    print("STEP 3: TRAINING MODELS")
    print("-" * 80)
    trainer = CICModelTrainer(data_path=preprocessed_data_path, model_dir=model_dir)
    trainer.load_data()
    trainer.split_data(test_size=test_size, min_samples=min_samples)

    if model == 'all':
        trainer.train_all_models(use_smote=(not no_smote), tune_hyperparams=(not no_tune))
    else:
        trainer.train_model(
            model_name=model,
            use_smote=(not no_smote),
            tune_hyperparams=(not no_tune),
            n_iter=n_iter,
        )

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("Outputs:")
    print(f"  - Cleaned data:        {cleaned_data_path}")
    print(f"  - Preprocessed data:   {preprocessed_data_path}")
    print(f"  - Artifacts dir:       {artifacts_dir}")
    print(f"  - Models dir:          {model_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Colab-friendly CIC training wrapper",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # IO and environment
    parser.add_argument("--input", type=str, required=True, help="Path to CIC dataset (CSV/Parquet) or directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--dataset-type", type=str, default="CIC-IDS2017", help="Dataset type label")
    parser.add_argument("--mount-drive", action="store_true", help="Mount Google Drive at /content/drive")
    parser.add_argument("--install-req", action="store_true", help="Install requirements.txt (Colab)")

    # Preprocessing
    parser.add_argument("--binary", action="store_true", help="Use binary labels (BENIGN vs ATTACK)")
    parser.add_argument("--sample-frac", type=float, default=None, help="Sample fraction for quick runs (0-1)")
    parser.add_argument("--feature-selection", type=str, default="variance", choices=["variance", "correlation", "all"], help="Feature selection method")
    parser.add_argument("--n-features", type=int, default=None, help="Number of features to select")
    parser.add_argument("--balance", action="store_true", help="Balance classes")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples per class")

    # Training
    parser.add_argument("--model", type=str, default="all", choices=["all", "random_forest", "xgboost", "logistic_regression", "gradient_boosting", "knn", "decision_tree"], help="Which model to train")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (0-1)")
    parser.add_argument("--n-iter", type=int, default=10, help="Number of iterations for hyperparameter search")
    parser.add_argument("--min-samples", type=int, default=2, help="Minimum samples required per class (default: 2)")
    
    args = parser.parse_args()
    
    if not (0 < args.test_size < 1):
        parser.error("--test-size must be in (0, 1)")

    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        binary=args.binary,
        sample_frac=args.sample_frac,
        feature_selection=args.feature_selection,
        n_features=args.n_features,
        balance=args.balance,
        max_samples=args.max_samples,
        model=args.model,
        no_smote=args.no_smote,
        no_tune=args.no_tune,
        test_size=args.test_size,
        n_iter=args.n_iter,
        min_samples=args.min_samples,
        mount_drive=args.mount_drive,
        install_req=args.install_req,
    )


if __name__ == "__main__":
    main()
