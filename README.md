# churn-passation

A production-ready ML pipeline for customer churn prediction.

## Project Structure

```
churn-passation/
├── config/
│   └── config.yaml          # Pipeline configuration
├── data/
│   ├── raw/                 # Raw input data
│   └── processed/           # Preprocessed data
├── outputs/
│   ├── models/              # Saved model artifacts
│   └── evaluation/          # Evaluation reports and plots
├── src/
│   ├── data/
│   │   ├── ingestion.py     # Data loading utilities
│   │   └── preprocessing.py # Data cleaning and preparation
│   ├── features/
│   │   └── engineering.py   # Feature engineering
│   ├── models/
│   │   ├── train.py         # Model training
│   │   └── evaluate.py      # Model evaluation
│   └── pipeline/
│       └── pipeline.py      # End-to-end pipeline orchestrator
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_model.py
├── main.py                  # CLI entry point
└── requirements.txt
```

## Pipeline Architecture

```
Raw Data
   │
   ▼
Data Ingestion (src/data/ingestion.py)
   │  • Load CSV / DataFrame
   │  • Validate schema
   │
   ▼
Data Preprocessing (src/data/preprocessing.py)
   │  • Drop irrelevant columns
   │  • Handle missing values
   │  • Encode categorical features
   │  • Scale numerical features
   │
   ▼
Feature Engineering (src/features/engineering.py)
   │  • Derive new features
   │  • Select important features
   │
   ▼
Model Training (src/models/train.py)
   │  • Train/test split
   │  • Handle class imbalance (SMOTE)
   │  • Fit model (XGBoost / RF / LR / LightGBM)
   │  • Save model artifact
   │
   ▼
Model Evaluation (src/models/evaluate.py)
   │  • Accuracy, Precision, Recall, F1, ROC-AUC
   │  • Confusion matrix
   │  • Classification report
   │
   ▼
Outputs (outputs/)
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Place your raw CSV file at `data/raw/churn_data.csv`. The file must contain a
`Churn` column (binary: 0/1 or True/False) as the target variable.

### 3. Configure the pipeline

Edit `config/config.yaml` to match your dataset (column names, model choice,
hyperparameters, etc.).

### 4. Run the pipeline

```bash
python main.py
```

Alternatively, run individual stages:

```bash
# Training only
python main.py --mode train

# Evaluation only (requires a saved model)
python main.py --mode evaluate
```

### 5. Run tests

```bash
pytest tests/
```

## Configuration Reference

| Key | Description |
|-----|-------------|
| `data.raw_path` | Path to the raw CSV data file |
| `data.target_column` | Name of the churn target column |
| `data.test_size` | Fraction of data used for testing |
| `model.algorithm` | Algorithm to use (`xgboost`, `random_forest`, `logistic_regression`, `lightgbm`) |
| `evaluation.metrics` | List of metrics to compute |
| `output.model_path` | Where to save the trained model |
