# ReadmitIQ

**30-Day Medicare Hospital Readmission Risk Predictor**

An end-to-end ML system that predicts whether a Medicare beneficiary will be readmitted to the hospital within 30 days of discharge. Built on publicly available CMS data and deployed as a production-style system on AWS.

---

## Live Demo

- **React Frontend:** `[add Vercel URL after deployment]`
- **API Docs:** `[add ALB URL after deployment]/docs`

---

## Architecture

```
React Frontend (Vercel)
        ↓
FastAPI REST API (Docker → AWS ECS Fargate)
        ↓
XGBoost Model (loaded from AWS S3)
```

---

## Data

Built on the [CMS DE-SynPUF](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files) — synthetic Medicare fee-for-service claims data with the same structure as actual CMS production data. Free, public domain, no DUA required.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost, SHAP |
| API | FastAPI, Pydantic |
| Container | Docker |
| Cloud | AWS ECS Fargate, ECR, S3, CloudFront |
| Frontend | React, Tailwind CSS, Recharts |

---

## Project Structure

```
readmitiq/
├── notebooks/
│   ├── 01_data_download.ipynb        # Download and validate SynPUF data
│   ├── 02_eda.ipynb                  # Exploratory data analysis
│   ├── 03_feature_engineering.ipynb  # Feature pipeline and target variable
│   ├── 04_modeling.ipynb             # Train, evaluate, SHAP
│   └── 05_model_validation.ipynb     # Subgroup analysis and pre-deploy checks
├── src/
│   ├── 06_train.py                   # Production training script (CLI)
│   └── 07_predict.py                 # Inference module (used by API)
├── api/
│   └── 08_main.py                    # FastAPI application
├── infra/
│   └── 09_deploy.sh                  # AWS ECR + ECS deployment script
├── data/                             # gitignored — download via notebook 01
├── model/                            # gitignored — stored in S3
├── Dockerfile
├── requirements.txt
└── model_card.md
```

---

## Running Locally

```bash
# 1. Clone and set up environment
git clone https://github.com/yourusername/readmitiq
cd readmitiq
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download data (follow instructions in notebooks/01_data_download.ipynb)

# 3. Run notebooks in order: 01 → 02 → 03 → 04 → 05

# 4. Start the API
uvicorn api.08_main:app --reload --port 8000

# 5. Test it
curl http://localhost:8000/health
# Visit http://localhost:8000/docs for interactive API docs
```

---

## Running with Docker

```bash
docker build -t readmitiq .
docker run -p 8000:8000 readmitiq
```

---

## Model Performance

*See `data/processed/04_eval_metrics.json` after running notebook 04.*

| Metric | Value |
|---|---|
| AUC-ROC | TBD |
| Average Precision | TBD |
| Brier Score | TBD |

---

## Model Card

See [model_card.md](model_card.md) for full documentation including intended use, limitations, bias analysis, and production requirements.

---

## Author

Dr. Nikki — [LinkedIn](#) | [GitHub](#)
