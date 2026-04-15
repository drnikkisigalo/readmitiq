# ============================================================
# SCRIPT: api/08_main.py
# ------------------------------------------------------------
# PROJECT      : ReadmitIQ — 30-Day Medicare Readmission Risk Predictor
# AUTHOR       : Dr. Nikki
# CREATED      : 2026-04-08
# LAST UPDATED : 2026-04-08
#
# PURPOSE
# -------
# FastAPI application — the HTTP layer that wraps the ML model.
# This is what runs inside the Docker container on AWS ECS Fargate.
#
# When a request comes in:
#   1. FastAPI validates the request body against the PatientFeatures schema
#   2. The validated data is passed to the predict() function in 07_predict.py
#   3. The prediction result is returned as JSON
#
# USAGE
# -----
#   # Local development
#   uvicorn api.08_main:app --reload --port 8000
#
#   # Inside Docker
#   uvicorn api.08_main:app --host 0.0.0.0 --port 8000
#
# ENDPOINTS
# ---------
#   GET  /health         — Health check (used by AWS ALB)
#   GET  /               — API info
#   POST /predict        — Single patient prediction
#   POST /predict/batch  — Batch prediction (list of patients)
#
# INPUTS
# ------
#   POST body: JSON matching PatientFeatures schema
#
# OUTPUTS
# -------
#   JSON response matching PredictionResponse schema
# ============================================================

import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Make sure src/ is on the Python path so we can import 07_predict
sys.path.append(str(Path(__file__).parent.parent / "src"))
from predict_07 import load_model_artifacts, predict   # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------
# Lifespan handler — runs once at startup and shutdown.
# This is the modern FastAPI way to load resources once
# instead of loading them on every request.
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup, release at shutdown."""
    log.info("Loading ReadmitIQ model artifacts...")
    load_model_artifacts()
    log.info("ReadmitIQ API is ready to serve predictions.")
    yield
    log.info("ReadmitIQ API shutting down.")


# ---------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------
app = FastAPI(
    title="ReadmitIQ API",
    description=(
        "Predicts 30-day hospital readmission risk for Medicare patients. "
        "Trained on CMS DE-SynPUF synthetic data. Not validated for clinical use."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# CORS middleware — allows the React frontend to call this API.
# In production, replace \"*\" with your actual frontend domain.
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TODO: restrict to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request schema — Pydantic validates every incoming request.
# If any field is missing or the wrong type, FastAPI returns
# a 422 error automatically before our code even runs.
# ---------------------------------------------------------
class PatientFeatures(BaseModel):
    """Input features for a single patient readmission prediction."""

    # --- Claim-level features ---
    los: float = Field(..., ge=0, le=180,
                       description="Length of stay in days (from CLM_UTLZTN_DAY_CNT)")
    clm_pmt_amt_log: float = Field(..., ge=0,
                                   description="Log of total claim payment amount")
    per_diem_log: float = Field(..., ge=0,
                                description="Log of per diem pass-through amount")
    ddctbl_amt_log: float = Field(..., ge=0,
                                  description="Log of inpatient deductible paid by beneficiary")
    has_other_payer: int = Field(..., ge=0, le=1,
                                 description="1 if Medicare was not primary payer (dual eligible proxy)")
    n_diagnoses: int = Field(..., ge=0, le=15,
                             description="Number of ICD-9 diagnosis codes on claim")
    n_procedures: int = Field(..., ge=0, le=6,
                              description="Number of ICD-9 procedure codes on claim")
    drg_mdc: int = Field(..., ge=0, le=25,
                         description="Major Diagnostic Category derived from DRG code")

    # --- Patient demographics ---
    age_at_admission: int = Field(..., ge=0, le=115,
                                  description="Patient age in years at time of admission")
    bene_sex_ident_cd: int = Field(..., ge=1, le=2,
                                   description="Sex (1=Male, 2=Female)")
    bene_race_cd: int = Field(..., ge=1, le=6,
                              description="Race (1=White, 2=Black, 3=Other, 4=Asian, 5=Hispanic, 6=Native)")

    # --- Chronic condition flags ---
    n_chronic_conditions: int = Field(..., ge=0, le=15,
                                      description="Total count of active chronic condition flags")
    sp_chf: int = Field(..., ge=0, le=1, description="Congestive Heart Failure flag (0 or 1)")
    sp_diabetes: int = Field(..., ge=0, le=1, description="Diabetes flag (0 or 1)")
    sp_copd: int = Field(..., ge=0, le=1, description="COPD flag (0 or 1)")
    sp_chrnkidn: int = Field(..., ge=0, le=1, description="Chronic Kidney Disease flag (0 or 1)")
    sp_strketia: int = Field(..., ge=0, le=1, description="Stroke/TIA flag (0 or 1)")

    # --- Prior utilization ---
    prior_inpatient_cnt: int = Field(..., ge=0,
                                     description="Inpatient admissions in prior 12 months")

    @field_validator("has_other_payer", "sp_chf", "sp_diabetes",
                     "sp_copd", "sp_chrnkidn", "sp_strketia")
    @classmethod
    def validate_binary_flags(cls, v):
        """All binary flags must be exactly 0 or 1."""
        if v not in (0, 1):
            raise ValueError("Binary flags must be 0 or 1")
        return v

    def to_feature_dict(self) -> dict:
        """
        Convert Pydantic model to the feature dict expected by 07_predict.py.
        Keys must exactly match FEATURE_COLUMNS in model/04_feature_columns.json.
        Order is enforced inside 07_predict.py using the saved JSON column list.
        """
        return {
            "LOS"                  : self.los,
            "CLM_PMT_AMT_LOG"      : self.clm_pmt_amt_log,
            "PER_DIEM_LOG"         : self.per_diem_log,
            "DDCTBL_AMT_LOG"       : self.ddctbl_amt_log,
            "HAS_OTHER_PAYER"      : self.has_other_payer,
            "N_DIAGNOSES"          : self.n_diagnoses,
            "N_PROCEDURES"         : self.n_procedures,
            "DRG_MDC"              : self.drg_mdc,
            "AGE_AT_ADMISSION"     : self.age_at_admission,
            "BENE_SEX_IDENT_CD"    : self.bene_sex_ident_cd,
            "BENE_RACE_CD"         : self.bene_race_cd,
            "N_CHRONIC_CONDITIONS" : self.n_chronic_conditions,
            "SP_CHF"               : self.sp_chf,
            "SP_DIABETES"          : self.sp_diabetes,
            "SP_COPD"              : self.sp_copd,
            "SP_CHRNKIDN"          : self.sp_chrnkidn,
            "SP_STRKETIA"          : self.sp_strketia,
            "PRIOR_INPATIENT_CNT"  : self.prior_inpatient_cnt,
        }


# ---------------------------------------------------------
# Response schema
# ---------------------------------------------------------
class RiskFactor(BaseModel):
    """A single feature and its SHAP contribution to the prediction."""
    feature: str
    impact: float


class PredictionResponse(BaseModel):
    """Prediction result returned to the caller."""
    readmission_probability : float
    risk_tier               : str
    risk_tier_description   : str
    top_risk_factors        : List[RiskFactor]
    model_version           : str
    disclaimer              : str


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/")
def root():
    """API info endpoint."""
    return {
        "name"       : "ReadmitIQ API",
        "version"    : "1.0.0",
        "description": "30-day Medicare readmission risk predictor",
        "docs"       : "/docs",
    }


@app.get("/health")
def health():
    """
    Health check endpoint.
    AWS ALB pings this every 30 seconds to confirm the container is alive.
    Returns 200 if the model is loaded and ready.
    """
    return {"status": "healthy", "model_version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict_single(patient: PatientFeatures):
    """
    Predict 30-day readmission risk for a single patient.

    Accepts a JSON body matching the PatientFeatures schema.
    Returns a readmission probability (0-1), risk tier, and top SHAP factors.
    """
    try:
        # Convert Pydantic model to the feature dict the predict function expects
        feature_dict = patient.to_feature_dict()

        # Run inference
        result = predict(feature_dict)

        return result

    except ValueError as e:
        # Invalid input values — return a 422 with the error detail
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        # Model not loaded — return a 503
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("Unexpected error during prediction: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(patients: List[PatientFeatures]):
    """
    Predict 30-day readmission risk for a list of patients.

    Accepts a JSON array of PatientFeatures objects.
    Returns an array of PredictionResponse objects in the same order.

    Useful for processing an entire discharge list at once.
    """
    if len(patients) > 500:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 500 patients per request."
        )

    results = []
    for i, patient in enumerate(patients):
        try:
            result = predict(patient.to_feature_dict())
            results.append(result)
        except Exception as e:
            log.error("Error on batch item %d: %s", i, str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed for item {i}: {str(e)}"
            )

    return results