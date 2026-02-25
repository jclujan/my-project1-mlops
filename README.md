# 🏠 Smart Residential Price Estimation  
### Production-Grade House Price Prediction – Ames, Iowa  

**Authors:** Group 3 – Juan Camilo Luján, Laurenz Jakob, Raluca Gogosoiu, Silvia Mendoza & Stephan Pentchev  
**Course:** MLOps – Master in Business Analytics and Data Science  
**Status:** Production Refactoring Phase  

---

# 1. Business Objective

## The Goal

Develop a scalable machine learning system that accurately predicts residential house sale prices using structured property data.

The solution aims to:

- Improve valuation accuracy  
- Reduce reliance on manual appraisal heuristics  
- Provide consistent, data-driven pricing  
- Enable scalable deployment from Ames (Iowa) to state-wide and eventually nationwide markets  

The long-term vision is to build a fully automated housing valuation engine for real estate agencies, lenders, and investors.

---

## The User

Primary users:

- Real estate agencies  
- Mortgage lenders  
- Property investors  
- Individual sellers  

Outputs:

- Automated sale price predictions  
- Evaluation reports  
- Scalable production-ready pipeline  

---

# 2. Success Metrics

## Business KPI (The “Why”)

- Reduce pricing error compared to manual estimation  
- Decrease time-to-price by 30%  
- Increase pricing consistency across neighborhoods  
- Improve listing competitiveness  

---

## Technical Metric (The “How”)

Model: **LassoCV (Regularized Linear Regression)**  

Validation Results:

- R² (train) ≈ 0.904  
- R² (validation) ≈ 0.893  
- 163 / 256 coefficients retained (automatic feature selection)  

Target:

- R² ≥ 0.88 on validation  
- Stable residual distribution  
- Controlled bias across price segments  

---

# 3. The Data

## Dataset Overview

Residential housing data from **Ames, Iowa**.

- Target Variable: `SalePrice` (USD)  
- 81 structured features including:
  - Living area (GrLivArea)  
  - Neighborhood  
  - Year built  
  - Basement and garage features  
  - Overall quality indicators  

---

## Preprocessing Pipeline

1. Missing Value Handling  
   - Categorical → "None"  
   - Numerical → 0  

2. Rare Category Grouping  
   - Categories with <10 observations grouped into "Other"  

3. Log Transformation  
   - Applied to skewed features  
   - Target variable modeled in log-space  

4. Multicollinearity Reduction  
   - Removed features with correlation > 0.8  

5. One-Hot Encoding  
   - Applied to categorical variables  

---

## Data Governance

- No personal identifiable information (PII)  
- `/data`, `/models`, `/reports` excluded from Git  

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                # This file (Project definition)
├── environment.yml          # Dependencies (Conda/Pip)
├── config.yaml              # Global configuration (paths, params)
├── .env                     # Secrets placeholder
│
├── notebooks/               # Experimental sandbox
│   └── yourbaseline.ipynb   # From previous work
│
├── src/                     # Production code (The "Factory")
│   ├── __init__.py          # Python package
│   ├── load_data.py         # Ingest raw data
│   ├── clean_data.py        # Preprocessing & cleaning
│   ├── validate.py          # Data quality checks
│   ├── train.py             # Model training & saving
│   ├── evaluate.py          # Metrics & plotting
│   ├── infer.py             # Inference logic
│   └── main.py              # Pipeline orchestrator
│
├── data/                    # Local storage (IGNORED by Git)
│   ├── raw/                 # Immutable input data
│   └── processed/           # Cleaned data ready for training
│
├── models/                  # Serialized artifacts (IGNORED by Git)
│
├── reports/                 # Generated metrics, plots, and figures
│
└── tests/                   # Automated tests
```

## 5. Execution Model

The full machine learning pipeline will eventually be executable through:

`python src/main.py`



