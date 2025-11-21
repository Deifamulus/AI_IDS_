# AI-Powered Network Intrusion Detection System (IDS)

A sophisticated Network Intrusion Detection System that combines Suricata's rule-based detection with advanced machine learning for comprehensive network security. The system analyzes network traffic in real-time, using both signature-based and behavior-based detection to identify and mitigate potential threats.

## Features

### Core Capabilities
- **Real-time Network Monitoring**: Actively monitors network traffic on specified interfaces
- **Suricata Integration**: Utilizes Suricata's powerful rule-based detection engine
- **AI-Powered Detection**: Employs machine learning models for advanced threat identification
- **Behavioral Analysis**: Detects anomalies and zero-day attacks using statistical models
- **Multi-class Classification**: Identifies specific attack types (DDoS, Brute Force, XSS, etc.)

### ML Models
- **XGBoost**: High-performance gradient boosting for accurate attack classification
- **Random Forest**: Ensemble learning for robust detection across various attack types
- **Logistic Regression**: Fast and efficient baseline model for binary classification
- **Support Vector Machines (SVM)**: Effective for high-dimensional feature spaces
- **Neural Networks**: Deep learning models for complex pattern recognition

### Additional Features
- **Web Dashboard**: Visual interface for monitoring alerts and network activity
- **Whitelisting**: Configurable IP and port whitelisting to reduce false positives
- **Logging**: Comprehensive logging of all network events and alerts
- **Model Retraining**: Tools for continuous improvement of detection capabilities

## Prerequisites

- Python 3.8+
- Suricata IDS
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai_ids_project
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   
   Note: For GPU acceleration (recommended for training):
   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
   ```

3. Install Suricata:
   - On Ubuntu/Debian: `sudo apt-get install suricata`
   - On CentOS/RHEL: `sudo yum install suricata`
   
4. Download pre-trained models or train your own (see [Training ML Models](#training-ml-models) section)

## Configuration

1. **Basic Setup**
   - Update Suricata rules directory in the configuration if needed
   - Configure whitelisted IPs and ports in `src/live_sniffing_improved.py`
   - Adjust alert sensitivity and monitoring parameters as needed

2. **AI/ML Configuration**
   - Model selection and thresholds in `config/model_config.json`
   - Feature extraction parameters in `src/feature_extractor.py`
   - Training parameters in `src/train_cic_model.py`

3. **Performance Tuning**
   - Adjust batch sizes for your hardware in `src/live_sniffing_improved.py`
   - Enable/disable specific ML models based on performance requirements
   - Configure model retraining schedule

## Usage

### Starting the IDS

```bash
python src/live_sniffing_improved.py -i <interface> [--rules-dir /path/to/rules] [--model-path /path/to/model.pkl] [--enable-ml]
```

Example with ML enabled:
```bash
python src/live_sniffing_improved.py -i eth0 --enable-ml --model-path models/xgboost_cic_model.pkl
```

### Training ML Models

1. **Using Preprocessed Data**
   ```bash
   python src/train_cic_model.py --data-path data/processed/cic_flows.csv --model-dir models/
   ```

2. **With Custom Parameters**
   ```bash
   python src/train_cic_model.py --data-path data/processed/cic_flows.csv \
                               --model-dir models/ \
                               --model xgboost \
                               --epochs 100 \
                               --batch-size 1024 \
                               --enable-gpu
   ```

### Accessing the Web Dashboard

1. Start the dashboard server:
   ```bash
   python dashboard.py
   ```
2. Open a web browser and navigate to: `http://localhost:5000`

### Model Evaluation

To evaluate a trained model:
```bash
python src/evaluate_model.py --model-path models/xgboost_cic_model.pkl --test-data data/processed/test_flows.csv
```

## Project Structure

```
ai_ids_project/
├── src/                           # Source code
│   ├── live_sniffing_improved.py  # Main IDS implementation
│   ├── train_cic_model.py         # ML model training pipeline
│   ├── feature_extractor.py       # Feature extraction utilities
│   ├── data_utils.py             # Data preprocessing and utilities
│   ├── models/                   # Custom model architectures
│   │   ├── xgboost_model.py
│   │   ├── random_forest.py
│   │   └── neural_network.py
│   └── ...                       # Other source files
│
├── config/                       # Configuration files
│   ├── model_config.json        # ML model configurations
│   └── suricata/                # Suricata configuration
│
├── data/                         # Data directories
│   ├── raw/                     # Raw network captures
│   ├── processed/               # Processed datasets
│   └── cache/                   # Cached features
│
├── models/                       # Trained ML models
│   ├── xgboost_cic_model.pkl
│   ├── random_forest_cic.pkl
│   └── model_metrics/           # Training metrics and plots
│
├── notebooks/                    # Jupyter notebooks
│   ├── data_analysis.ipynb
│   ├── model_exploration.ipynb
│   └── visualization.ipynb
│
├── requirements.txt              # Python dependencies
└── dashboard.py                 # Web dashboard
```

## AI/ML Customization

### Model Training
- **Data Preparation**
  - Use `src/data_utils.py` to preprocess your network traffic data
  - Ensure proper feature engineering for optimal model performance

- **Training New Models**
  - Configure training parameters in `config/model_config.json`
  - Use `src/train_cic_model.py` for training new models
  - Enable GPU acceleration for faster training with `--enable-gpu`

- **Model Evaluation**
  - Evaluate model performance using precision, recall, and F1-score
  - Analyze confusion matrices and ROC curves
  - Monitor model drift and retrain as needed

### Integration
- **Custom Detectors**
  - Implement custom detection algorithms in `src/models/`
  - Register new models in `src/model_factory.py`
  
- **Feature Engineering**
  - Modify `src/feature_extractor.py` to add new features
  - Implement custom feature selection strategies

### Performance Tuning
- **Inference Optimization**
  - Adjust batch sizes for your hardware
  - Enable model quantization for faster inference
  - Implement model distillation for resource-constrained environments

## License

[Specify your license here]

## Contributing

We welcome contributions to improve the AI/ML capabilities of this IDS! Here's how you can help:

1. **Model Improvements**
   - Implement new ML architectures
   - Add support for transfer learning
   - Improve feature engineering

2. **Performance Optimization**
   - Optimize inference speed
   - Reduce memory footprint
   - Add support for distributed training

3. **New Features**
   - Implement new detection techniques
   - Add support for additional network protocols
   - Create new visualization tools

Please submit your contributions via Pull Requests.

## Research and References

This project builds upon several research papers and open-source projects:

- **CIC-IDS Dataset**: For training and evaluation
- **XGBoost**: Gradient boosting framework
- **Scikit-learn**: Machine learning in Python
- **Suricata**: Network threat detection engine
- **Scapy**: Packet manipulation library

## Citing This Work

If you use this project in your research, please consider citing:

```
@misc{ai_ids_project,
  author = {Your Name},
  title = {AI-Powered Network Intrusion Detection System},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/ai_ids_project}}
}
```