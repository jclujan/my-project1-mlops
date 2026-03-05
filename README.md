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

New pipeline:

- R² (train) ≈ 0.804 

Notebook before:

Validation Results:

- R² (train) ≈ 0.904  
- R² (validation) ≈ 0.893  
- 163 / 256 coefficients retained (automatic feature selection)  

Target:

- R² ≥ 0.8 
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

This project follows a strict separation between experimental work, production code, data layers, and testing.

```text
.
├── README.md                  # Project documentation
├── LICENSE                    # License file
├── .gitignore                 # Git ignored files
├── config.yaml                # Global configuration (paths, parameters)
├── environment.yml            # Conda environment specification
│
├── data/                      # Local data storage (ignored by Git)
│   ├── raw/                   # Immutable original datasets
│   ├── processed/             # Cleaned datasets for training
│   └── inference/             # Unseen data for inference (no SalePrice)
│
├── models/                    # Saved trained model artifacts (ignored by Git)
│
├── reports/                   # Generated metrics, evaluation outputs, predictions
│
├── notebook/                  # Experimental sandbox (Jupyter notebooks)
│   ├── HousePred-LassoReg.ipynb   # Original baseline notebook
│   ├── experiment.pynb       # notebook for data scientists (experimental)
│
├── src/                       # Production pipeline (core ML system)
│   ├── __init__.py            # Makes src a Python package
│   ├── clean_data.py          # Data preprocessing & feature preparation
│   ├── evaluate.py            # Model evaluation and metrics handling
│   ├── features.py            # Feature engineering utilities
│   ├── infer.py               # Inference logic (prediction on new data)
│   ├── load_data.py           # Data loading utilities
│   ├── main.py                # Pipeline orchestrator (training workflow)
│   ├── train.py               # Model training logic
│   ├── utils.py               # Helper utilities (e.g., path handling)
│   └── validate.py            # Data validation checks
│
└── tests/                     # Automated unit tests (pytest)
```

## 5. Execution Model & Environment setup

The full machine learning pipeline will eventually be executable through:

1.⁠ ⁠environment setup:
`conda env create -f environment.yml`
`conda activate mlops_project`

2.⁠ ⁠launch sandbox:
`code notebook/HousePred-LassoReg.ipynb`

3.⁠ ⁠test suite:
`python -m pytest -q`

4.⁠ ⁠orchestrator:
`python -m src.main`

## 6. Generated outputs

1.⁠ ⁠data/processed/clean.csv: The deterministically cleaned input data

2.⁠ ⁠models/model.joblib: The deployable pipeline artifact

3.⁠ ⁠reports/predictions.csv: The inference log containing predictions 


---

# Business Overview and Understanding: Canvas

## Client

Real Estate Agency / Property Valuation Firm  

Industry: Real Estate & Financial Services  

---

## Business Unit

- Property Valuation  
- Sales Strategy  
- Investment Analysis  

---

## Client Maturity

- Structured housing data available  
- Spreadsheet-based pricing workflow  
- Manual appraisal processes  
- Limited automation  
- Strong interest in analytics-driven valuation  

---

## Problem Statement

Current pricing relies heavily on:

- Manual valuation  
- Comparable property heuristics  
- Subjective experience  

### Pain Points

- Inconsistent pricing  
- Bias in high-end property estimates  
- Time-consuming valuation process  
- Missed revenue opportunities  

---

## Goal (Quantifiable KPI)

- Validation R² ≥ 0.8  
- Reduced pricing variance  
- Faster listing preparation  
- Improved pricing consistency  

---

## Solution Description & Key Functionalities

A scalable machine learning valuation engine that:

- Cleans and preprocesses structured housing data  
- Identifies key value drivers  
- Produces automated sale price predictions  
- Ensures reproducible and modular MLOps pipeline execution  

### Future Extensions

- REST API deployment  
- Real-time pricing dashboard  
- Multi-region training framework  

---

## Solution Scalability

### Short-Term
- Production model for Ames  

### Mid-Term
- Expand to Iowa statewide data  
- Incorporate regional economic indicators  

### Long-Term
- Nationwide automated valuation engine  
- Integration with real estate listing platforms  
- Continuous retraining and monitoring  

---

## Client Benefits (vs Non-AI Approach)

### Tangible Benefits

- Higher pricing accuracy  
- Reduced manual workload  
- Faster time-to-market  
- More consistent pricing strategy  

### Intangible Benefits

- Data-driven credibility  
- Competitive advantage  
- Scalable analytics infrastructure  

### Measured By

- Reduced prediction error  
- Improved R²  
- Reduced listing cycle time  

---

## Cost Estimation (Ballpark)

### Talent Requirements

- AI Specialist  
- ML Engineer  
- Data Engineer  
- Product Manager  
- Real Estate Subject Matter Expert  

Estimated development phase: **12+ weeks**

Estimated pilot budget: **$60k – $120k**

### Client Responsibilities

- Data infrastructure  
- Cloud compute  
- Deployment environment  

---

## Risks & Mitigation

### Risks

- Limited generalization beyond Ames  
- Bias at price extremes  
- Data quality variations across regions  
- Model drift over time  

### Mitigation

- Continuous monitoring of residuals  
- Scheduled retraining  
- Bias audits  
- Geographic data expansion  