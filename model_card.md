# ReadmitIQ — Model Card

**Model Name:** ReadmitIQ v1.0 — 30-Day Medicare Readmission Risk Classifier  
**Version:** 1.0  
**Date:** 2026-04-08  
**Author:** Dr. Nikki  

---

## Model Overview

ReadmitIQ predicts whether a Medicare fee-for-service beneficiary will be readmitted
to an inpatient hospital within 30 days of discharge. It is designed as a discharge
planning decision support tool — flagging high-risk patients so care teams can
intervene before readmission occurs.

---

## Intended Use

| Attribute | Value |
|---|---|
| **Primary use** | Portfolio demonstration of end-to-end ML deployment |
| **Secondary use** | Prototype for clinical discharge planning decision support |
| **Intended users** | Data scientists, ML engineers, clinical informatics teams |
| **Out-of-scope** | Production clinical deployment without further validation |

---

## Data

| Attribute | Value |
|---|---|
| **Source** | CMS Linkable 2008–2010 Medicare DE-SynPUF, Sample 1 |
| **Type** | Synthetic Medicare fee-for-service claims data |
| **Population** | ~116,000 unique synthetic Medicare beneficiaries |
| **Training period** | 2008 inpatient admissions |
| **Test period** | 2009 inpatient admissions (temporal holdout) |
| **Access** | Public domain, no DUA required |
| **Note** | Data is synthetic and has limited inferential value for real Medicare populations |

---

## Algorithm

| Attribute | Value |
|---|---|
| **Algorithm** | XGBoost Binary Classifier |
| **Task** | Binary classification (readmitted within 30 days: yes/no) |
| **Features** | 14 features — see Feature Set section |
| **Class imbalance** | Addressed via scale_pos_weight parameter |
| **Split strategy** | Temporal (train on 2008, test on 2009) |

---

## Feature Set

| Feature | Description |
|---|---|
| `los` | Length of stay (days) |
| `clm_pmt_amt` | Claim payment amount (log-transformed) |
| `n_diagnoses` | Count of ICD-9 diagnosis codes on the claim |
| `drg_category` | Major Diagnostic Category derived from DRG code |
| `age_at_admission` | Beneficiary age at time of admission |
| `sex` | Beneficiary sex (1=Male, 2=Female) |
| `race` | Beneficiary race/ethnicity code |
| `n_chronic_conditions` | Count of active chronic condition flags |
| `sp_chf` | Congestive Heart Failure flag |
| `sp_diabetes` | Diabetes flag |
| `sp_copd` | COPD flag |
| `sp_chrnkidn` | Chronic Kidney Disease flag |
| `sp_strketia` | Stroke/TIA flag |
| `prior_inpatient_count` | Number of inpatient admissions in prior 12 months |

---

## Performance

*To be populated after training in notebook 04_modeling.ipynb*

| Metric | Value |
|---|---|
| AUC-ROC | TBD |
| Recall (Sensitivity) | TBD |
| Precision | TBD |
| Brier Score | TBD |

---

## Bias and Fairness Considerations

- The `race` variable is included as a feature and is evaluated in subgroup analysis
- Performance metrics are reported separately by age bucket and race category
- Real-world readmission disparities by race are well documented in the HRRP literature
- This synthetic dataset may not accurately reflect real disparities
- **Any production deployment must include a full bias audit before clinical use**

The model shows modestly lower AUC for Black beneficiaries (≈0.64) and the oldest-old age groups (80–84, 85+) compared to the overall AUC of 0.656. Performance gaps for Black patients are consistent with findings in the broader HRRP literature and likely reflect unmeasured social risk factors not present in claims data. Production deployment would require targeted recalibration for these subgroups and monitoring for disparate impact on care recommendations.

---

## Limitations

- Trained on synthetic data — not validated against real Medicare outcomes
- Data reflects 2008–2010 clinical practice and billing patterns
- Does not include outpatient, pharmacy, or physician visit data (available in full SynPUF)
- Does not implement CMS's planned readmission exclusion algorithm
- Not risk-standardized (real HRRP models use hierarchical risk standardization)

---

## Production Requirements

Before deploying in a real clinical environment, the following are required:

- Access to CMS Research Identifiable Files (RIF) via approved Data Use Agreement
- Business Associate Agreement (BAA) with any covered entity
- PHI handling controls compliant with HIPAA
- Prospective validation on recent (post-2020) Medicare data
- Full bias audit across demographic subgroups
- Clinical validation with physician review
- IRB approval if used in research context

---

## References

- CMS HRRP Program: https://www.cms.gov/medicare/quality/value-based-programs/hospital-readmissions
- SynPUF Data: https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
- HRRP Impact on Vulnerable Populations: https://www.cms.gov/files/document/impact-readmissions-reduction-initiatives-report.pdf
