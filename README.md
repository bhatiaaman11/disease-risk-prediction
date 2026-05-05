# Disease Risk Prediction System

A machine learning pipeline for predicting heart disease risk from patient health indicators - built as a proof of concept for a generalizable, disease-agnostic clinical decision support framework.

---

## Project Overview

Early disease detection is critical for improving patient outcomes and reducing long-term healthcare costs. This project implements an end-to-end ML pipeline that processes raw patient data, trains and evaluates multiple classification models, and delivers interpretable risk predictions with clinical relevance.

The system is designed to be **disease-agnostic** - heart disease prediction using the UCI Heart Disease dataset serves as the initial proof of concept, with the framework extensible to other conditions such as diabetes or stroke.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.87 |
| Recall | 0.93 |
| F1 Score | 0.89 |
| ROC-AUC | 0.92 |

> **Final model: Logistic Regression** - selected over tuned ensemble alternatives (Random Forest, Gradient Boosting, XGBoost) based on superior interpretability, balanced performance, and stability (CV std = 0.021).
>
> Threshold optimization (0.50 → 0.45) reduced missed diagnoses by improving recall from 88% to 93% while addressing a sex-based subgroup performance gap - with no increase in false alarms.

---

## Pipeline Architecture

The project is structured across 8 Jupyter notebooks, each covering a distinct stage of the ML pipeline:

| Notebook | Stage |
|----------|-------|
| `01_data_loading` | Data loading and inspection |
| `02_preprocessing` | KNN imputation, encoding, Winsorization, scaling, train/val/test split |
| `03_eda` | Exploratory data analysis - distributions, correlations, outlier detection |
| `04_baseline_model` | Logistic Regression baseline with cross-validation |
| `05_advanced_models` | Random Forest, Gradient Boosting, XGBoost with GridSearchCV tuning |
| `06_evaluation` | Test set evaluation, ROC curves, subgroup analysis, threshold optimization |
| `07_explainability` | SHAP-based feature importance and clinical insight |
| `08_prediction_workflow` | End-to-end prediction for raw patient input dictionaries |

---

## Key Technical Decisions

- **KNN Imputation** over mean/mode imputation - preserves relationships between features for missing values
- **Retained high-missingness features** (`ca`, `thal`) using missing indicators + KNN imputation rather than dropping them, these features carry strong clinical signal
- **Label encoding with manual mappings** over one-hot encoding - clinically defensible severity gradients (especially `thal`), and appropriate for small dataset size (~920 rows)
- **Winsorization on cholesterol** - handles outliers without data loss
- **Stratified 70/15/15 split** - ensures consistent class distribution across train, validation, and test sets
- **Recall-weighted model selection** - in clinical contexts, missed diagnoses (false negatives) carry higher cost than false alarms

---

## Explainability

SHAP (SHapley Additive exPlanations) was used to provide model-agnostic interpretability:
- Summary, bar, waterfall, and dependence plots generated for the final model
- Key features influencing predictions identified and visualized
- Clinical reasoning behind predictions made transparent for end users

---

## Tech Stack

- **Language:** Python 3
- **ML:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebooks, Anaconda

---

## Dataset

**UCI Heart Disease Dataset** - sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)

> The dataset is not included in this repository. Download it directly from the UCI repository and place it in `data/raw/`.

---

## Project Structure

```
disease-risk-prediction/
│
├── notebooks/          # 8 Jupyter notebooks covering the full ML pipeline
├── outputs/
│   └── figures/        # Generated charts and plots
├── docs/               # Project proposal and final report (PDF)
└── .gitignore
```

---

## Authors

**Amanpreet Bhatia**
Graduate students - Northeastern University 
Course - Data Science Engineering with Python (DAMG6105)
