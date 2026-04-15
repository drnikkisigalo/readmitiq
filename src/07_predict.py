# ============================================================
# SCRIPT: 07_predict.py
# ------------------------------------------------------------
# PROJECT      : ReadmitIQ — 30-Day Medicare Readmission Risk Predictor
# AUTHOR       : Dr. Nikki
# CREATED      : 2026-04-08
# LAST UPDATED : 2026-04-08
#
# PURPOSE
# -------
# Inference module — contains all the logic needed to take a
# raw patient input dictionary and return a readmission risk score.
#
# This module is imported by the FastAPI service (08_main.py).
# It is NOT run directly. Keeping inference logic here (separate
# from the API layer) makes it easy to test and reuse independently.
#
# DESIGN NOTES
# ------------
# - Loads the model from disk OR from S3 depending on environment
# - Handles feature ordering automatically using the saved JSON
# - Returns top SHAP contributors so the API can explain predictions
# - All functions are stateless except load_model_artifacts()
#
# INPUTS
# ------
#   model/04_readmitiq_xgb_v1.pkl    (or S3 path via env vars)
#   model/04_feature_columns.json
#   model/04_shap_explainer.pkl
#
# OUTPUTS
# -------
#   This module exports: load_model_artifacts(), predict()
# ============================================================

import os
import json
import logging
import joblib
import boto3
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------
# Module-level cache for model artifacts.
# We load once at startup and reuse for every request.
# Loading a model on every request would be very slow.
# ---------------------------------------------------------
_MODEL     = None
_EXPLAINER = None
_FEATURES  = None


def load_model_artifacts(
    model_path: Optional[str] = None,
    features_path: Optional[str] = None,
    shap_path: Optional[str] = None,
) -> None:
    """
    Load the trained model, feature columns, and SHAP explainer into memory.

    Checks two locations in order:
    1. Local file paths (for local development and testing)
    2. AWS S3 (for production deployment on ECS Fargate)

    The S3 bucket and key are read from environment variables:
        MODEL_S3_BUCKET — S3 bucket name
        MODEL_S3_KEY    — Path to model .pkl file in the bucket

    Call this once at application startup.
    """
    global _MODEL, _EXPLAINER, _FEATURES

    # Determine paths — prefer explicit arguments, then defaults
    script_dir   = Path(__file__).parent
    project_root = script_dir.parent
    model_dir    = project_root / "model"

    model_path    = model_path    or str(model_dir / "04_readmitiq_xgb_v1.pkl")
    features_path = features_path or str(model_dir / "04_feature_columns.json")
    shap_path     = shap_path     or str(model_dir / "04_shap_explainer.pkl")

    # --- Load feature columns (always from local path) ---
    if Path(features_path).exists():
        with open(features_path) as f:
            _FEATURES = json.load(f)
        log.info("Feature columns loaded: %d features", len(_FEATURES))
    else:
        raise FileNotFoundError(f"Feature columns file not found: {features_path}")

    # --- Load model ---
    s3_bucket = os.getenv("MODEL_S3_BUCKET")
    s3_key    = os.getenv("MODEL_S3_KEY")

    if Path(model_path).exists():
        # Load from local disk (development)
        log.info("Loading model from local path: %s", model_path)
        _MODEL = joblib.load(model_path)
    elif s3_bucket and s3_key:
        # Download from S3 to a temp file (production on ECS)
        log.info("Loading model from S3: s3://%s/%s", s3_bucket, s3_key)
        s3 = boto3.client("s3")
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            s3.download_fileobj(s3_bucket, s3_key, tmp)
            tmp_path = tmp.name
        _MODEL = joblib.load(tmp_path)
        Path(tmp_path).unlink()  # Clean up temp file
    else:
        raise FileNotFoundError(
            f"Model not found at {model_path} and S3 env vars not set. "
            "Set MODEL_S3_BUCKET and MODEL_S3_KEY environment variables for production."
        )

    log.info("Model loaded: %s", type(_MODEL).__name__)

    # --- Load SHAP explainer (optional — only needed for explanations) ---
    if Path(shap_path).exists():
        _EXPLAINER = joblib.load(shap_path)
        log.info("SHAP explainer loaded.")
    else:
        log.warning("SHAP explainer not found at %s. Predictions will work but explanations will not.", shap_path)


def _get_risk_tier(probability: float) -> tuple[str, str]:
    """
    Map a probability score to a human-readable risk tier and description.

    Returns (tier_label, tier_description)
    """
    if probability >= 0.5:
        return "High", "Patient has >50% predicted probability of 30-day readmission."
    elif probability >= 0.3:
        return "Moderate", "Patient has moderate predicted probability of 30-day readmission."
    else:
        return "Low", "Patient has low predicted probability of 30-day readmission."


def _get_top_shap_factors(input_df: pd.DataFrame, n_factors: int = 5) -> list[dict]:
    """
    Compute SHAP values for a single input and return the top contributing features.

    Returns a list of dicts with 'feature' and 'impact' keys,
    sorted by absolute impact descending.
    """
    if _EXPLAINER is None:
        return []

    try:
        shap_vals = _EXPLAINER.shap_values(input_df)
        factors = sorted(
            [
                {
                    "feature": feat,
                    "impact" : round(float(shap_vals[0][i]), 4)
                }
                for i, feat in enumerate(_FEATURES)
            ],
            key=lambda x: abs(x["impact"]),
            reverse=True
        )
        return factors[:n_factors]
    except Exception as e:
        log.warning("SHAP computation failed: %s", str(e))
        return []


def predict(patient_features: dict) -> dict:
    """
    Run inference on a single patient's features.

    Args:
        patient_features: dict mapping feature name to value.
                          Must contain all keys in FEATURE_COLUMNS.

    Returns:
        dict with readmission_probability, risk_tier, risk_tier_description,
        top_risk_factors, model_version, disclaimer.

    Raises:
        RuntimeError if model artifacts have not been loaded.
        ValueError if required feature columns are missing.
    """
    if _MODEL is None or _FEATURES is None:
        raise RuntimeError(
            "Model artifacts not loaded. Call load_model_artifacts() at startup."
        )

    # ---------------------------------------------------------
    # Validate that all required feature columns are present
    # ---------------------------------------------------------
    missing = [col for col in _FEATURES if col not in patient_features]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # ---------------------------------------------------------
    # Build the input dataframe in the EXACT feature order
    # the model was trained on. Order matters for XGBoost.
    # ---------------------------------------------------------
    input_df = pd.DataFrame(
        [[patient_features[col] for col in _FEATURES]],
        columns=_FEATURES
    )

    # ---------------------------------------------------------
    # Run inference — predict_proba gives calibrated probabilities
    # We take column [1] = probability of class 1 (readmitted)
    # ---------------------------------------------------------
    probability = float(_MODEL.predict_proba(input_df)[0][1])
    probability = round(probability, 4)

    risk_tier, risk_description = _get_risk_tier(probability)

    # ---------------------------------------------------------
    # Get top SHAP contributors for explainability
    # ---------------------------------------------------------
    top_risk_factors = _get_top_shap_factors(input_df)

    return {
        "readmission_probability" : probability,
        "risk_tier"               : risk_tier,
        "risk_tier_description"   : risk_description,
        "top_risk_factors"        : top_risk_factors,
        "model_version"           : "1.0",
        "disclaimer"              : (
            "Trained on CMS DE-SynPUF synthetic data. "
            "Not validated for clinical use."
        ),
    }
