# ============================================================
# SCRIPT: 06_train.py
# ------------------------------------------------------------
# PROJECT      : ReadmitIQ — 30-Day Medicare Readmission Risk Predictor
# AUTHOR       : Dr. Nikki
# CREATED      : 2026-04-08
# LAST UPDATED : 2026-04-08
#
# PURPOSE
# -------
# Production training script — the cleaned, runnable version
# of the logic developed in notebooks 03 and 04.
#
# This script is NOT for exploration. It is designed to be
# run from the command line to retrain the model on demand,
# and is the script that would be triggered by an MLOps
# pipeline (e.g., a scheduled AWS Batch job or GitHub Action).
#
# USAGE
# -----
#   python src/06_train.py
#   python src/06_train.py --output-dir model/
#
# INPUTS
# ------
#   data/processed/03_X_train.csv
#   data/processed/03_y_train.csv
#   data/processed/03_X_test.csv
#   data/processed/03_y_test.csv
#   data/processed/03_feature_columns.json
#
# OUTPUTS
# -------
#   model/04_readmitiq_xgb_v1.pkl
#   model/04_feature_columns.json
#   model/04_shap_explainer.pkl
#   data/processed/04_eval_metrics.json
# ============================================================

import argparse
import json
import joblib
import logging
import shap
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# ---------------------------------------------------------
# Configure logging so we get clean, timestamped output
# instead of bare print statements
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_data(processed_dir: Path) -> tuple:
    """
    Load the pre-built feature matrices from notebook 03.

    Returns X_train, y_train, X_test, y_test, feature_columns
    """
    log.info("Loading feature matrices from %s", processed_dir)

    X_train = pd.read_csv(processed_dir / "03_X_train.csv")
    y_train = pd.read_csv(processed_dir / "03_y_train.csv").squeeze()
    X_test  = pd.read_csv(processed_dir / "03_X_test.csv")
    y_test  = pd.read_csv(processed_dir / "03_y_test.csv").squeeze()

    with open(processed_dir / "03_feature_columns.json") as f:
        feature_columns = json.load(f)

    log.info("X_train: %s rows, X_test: %s rows, Features: %d",
             f"{len(X_train):,}", f"{len(X_test):,}", len(feature_columns))

    return X_train, y_train, X_test, y_test, feature_columns


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_test: pd.DataFrame, y_test: pd.Series,
                feature_columns: list) -> XGBClassifier:
    """
    Train the XGBoost binary classifier with class imbalance handling.

    Uses scale_pos_weight to handle the imbalanced readmission target.
    Trains with early stopping evaluated on the test AUC.
    """
    # Calculate class weight for the minority class (readmitted=1)
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive

    log.info("Class balance — Negative: %d, Positive: %d, scale_pos_weight: %.2f",
             n_negative, n_positive, scale_pos_weight)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    log.info("Training XGBoost model...")
    start = datetime.now()

    model.fit(
        X_train[feature_columns],
        y_train,
        eval_set=[(X_test[feature_columns], y_test)],
        verbose=False,
    )

    elapsed = (datetime.now() - start).seconds
    log.info("Training complete in %ds.", elapsed)

    return model


def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame,
                   y_test: pd.Series, feature_columns: list,
                   n_train: int, n_test: int) -> dict:
    """
    Generate predictions and compute evaluation metrics.

    Returns a dictionary of metrics suitable for logging and the model card.
    """
    y_prob = model.predict_proba(X_test[feature_columns])[:, 1]

    auc_roc  = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    brier    = brier_score_loss(y_test, y_prob)

    log.info("Evaluation — AUC-ROC: %.4f | Avg Precision: %.4f | Brier: %.4f",
             auc_roc, avg_prec, brier)

    metrics = {
        "model"            : "XGBoost Binary Classifier",
        "version"          : "1.0",
        "train_year"       : 2008,
        "test_year"        : 2009,
        "n_train"          : int(n_train),
        "n_test"           : int(n_test),
        "n_features"       : len(feature_columns),
        "auc_roc"          : round(float(auc_roc), 4),
        "avg_precision"    : round(float(avg_prec), 4),
        "brier_score"      : round(float(brier), 4),
        "evaluated_at"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return metrics


def save_artifacts(model: XGBClassifier, feature_columns: list,
                   metrics: dict, X_test: pd.DataFrame,
                   output_dir: Path, processed_dir: Path) -> None:
    """
    Save the trained model, SHAP explainer, feature columns,
    and evaluation metrics to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "04_readmitiq_xgb_v1.pkl"
    joblib.dump(model, model_path)
    log.info("Model saved: %s", model_path)

    # Save feature column order — critical for inference consistency
    feat_path = output_dir / "04_feature_columns.json"
    with open(feat_path, "w") as f:
        json.dump(feature_columns, f, indent=2)
    log.info("Feature columns saved: %s", feat_path)

    # Compute and save SHAP explainer on a 2000-row sample
    log.info("Computing SHAP explainer (sample of 2000 rows)...")
    explainer  = shap.TreeExplainer(model)
    X_shap     = X_test[feature_columns].sample(min(2000, len(X_test)), random_state=42)
    _          = explainer.shap_values(X_shap)  # warm up the explainer
    shap_path  = output_dir / "04_shap_explainer.pkl"
    joblib.dump(explainer, shap_path)
    log.info("SHAP explainer saved: %s", shap_path)

    # Save evaluation metrics
    metrics_path = processed_dir / "04_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved: %s", metrics_path)


def main():
    """
    Entry point. Parse arguments and run the full training pipeline.
    """
    parser = argparse.ArgumentParser(description="Train the ReadmitIQ XGBoost model.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model",
        help="Directory to save model artifacts (default: model/)"
    )
    args = parser.parse_args()

    # Resolve all paths relative to this script's location
    script_dir    = Path(__file__).parent
    project_root  = script_dir.parent
    processed_dir = project_root / "data" / "processed"
    output_dir    = project_root / args.output_dir

    log.info("=== ReadmitIQ Training Pipeline ===")
    log.info("Project root  : %s", project_root)
    log.info("Output dir    : %s", output_dir)

    # Step 1: Load data
    X_train, y_train, X_test, y_test, feature_columns = load_data(processed_dir)

    # Step 2: Train model
    model = train_model(X_train, y_train, X_test, y_test, feature_columns)

    # Step 3: Evaluate
    metrics = evaluate_model(model, X_test, y_test, feature_columns, len(X_train), len(X_test))

    # Step 4: Save artifacts
    save_artifacts(model, feature_columns, metrics, X_test, output_dir, processed_dir)

    log.info("=== Training pipeline complete ===")


if __name__ == "__main__":
    main()
