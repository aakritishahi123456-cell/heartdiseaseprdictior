# ❤️ Heart Disease Risk Stratification System

A production-style machine learning system for predicting **cardiovascular risk** using structured clinical data.

This project goes beyond basic modeling by focusing on **data integrity, clinical reasoning, and decision thresholds**, making it closer to a real-world healthcare ML workflow.

---

## 🚀 Overview

This project formulates heart disease prediction as a **binary risk stratification problem**, not diagnosis.

The objective is to identify **high-risk patients early**, enabling further clinical evaluation.

Key characteristics:

- End-to-end ML pipeline
- Leakage-free preprocessing
- Clinical metric prioritization (recall)
- Threshold optimization for real-world tradeoffs
- Interactive inference via Streamlit

---

## 📊 Dataset

- Source: UCI Heart Disease Dataset
- Samples: **302 (after cleaning)**
- Features: 13 clinical variables
- Target:
  - `0 → Low Risk`
  - `1 → High Risk`

### ⚠️ Critical Data Issue (Key Insight)

The dataset initially appeared as **1025 rows**, but:

- **723 duplicate records were discovered**
- This artificially inflated performance

After removing duplicates:

- Dataset reduced to **302 rows**
- Model performance became realistic

👉 This was the most important engineering correction in the project.

---

## 🧠 Problem Framing

This is a **screening problem**, not a diagnostic system.

### Why this matters:

- False Negative → Miss a high-risk patient ❌ (dangerous)
- False Positive → Extra testing ✅ (acceptable)

👉 Therefore:

**Recall > Precision**

---

## ⚙️ ML Pipeline

### Steps:

1. Data loading and validation
2. Duplicate detection and removal
3. Exploratory Data Analysis (EDA)
4. Train/Test split
5. Preprocessing via `ColumnTransformer`
6. Model training:
   - Logistic Regression
   - KNN
   - Random Forest
7. Hyperparameter tuning
8. Threshold optimization
9. Model serialization (`.pkl`)
10. Streamlit deployment

---

## 🧩 Key Engineering Decisions

### 1. Duplicate Removal (Critical)
- Prevented **data leakage**
- Avoided inflated performance

### 2. Proper Feature Treatment
Categorical variables handled correctly:
- `cp`, `thal`, `slope`, `restecg`

### 3. Leakage-Free Pipeline
Used:
- `Pipeline`
- `ColumnTransformer`

Ensures:
- Train/test separation integrity

### 4. Metric Selection (Clinical Reasoning)
- Prioritized **Recall**
- Not accuracy (misleading in healthcare)

### 5. Threshold Optimization
Moved from:
- `0.5 → 0.4`

To improve sensitivity.

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| Logistic Regression | 0.836 | 0.848 | 0.848 | 0.848 |
| KNN | 0.836 | 0.871 | 0.818 | 0.844 |
| Random Forest | 0.803 | 0.818 | 0.818 | 0.818 |

---

## 🏆 Final Model

**Logistic Regression**

### Why:

- Best balance across metrics
- Stable after cleaning
- Interpretable (important in healthcare)
- Works well on small tabular datasets

---

## 🎯 Threshold Optimization

| Threshold | Recall | Precision |
|----------|--------|-----------|
| 0.3 | 0.970 | 0.653 |
| 0.4 | 0.939 | 0.795 |
| 0.5 | 0.848 | 0.848 |
| 0.6 | 0.636 | 0.913 |

### Final Choice:
**0.4**

👉 Maximizes detection of high-risk patients while keeping precision acceptable.

---

## 🖥️ Streamlit App

An interactive UI for real-time prediction.

### Features:

- Human-readable clinical inputs
- Probability-based output
- Threshold-aware classification
- Risk visualization
- Medical disclaimer

---

## ☁️ Supabase Setup

This project can save prediction records to Supabase through the implementation in `src/supabase_client.py`.

### 1. Create a Supabase project

1. Create a new project in Supabase.
2. Copy the project URL and `anon` key from the Supabase dashboard.
3. Set them as environment variables before starting Streamlit.

### 2. Set PowerShell environment variables

Use these commands in the current PowerShell session:

```powershell
$env:SUPABASE_URL = "https://YOUR-PROJECT-REF.supabase.co"
$env:SUPABASE_KEY = "YOUR-ANON-KEY"
```

If these variables are not set, the app still works locally, but prediction records will not be saved.

### 3. Create the `public.predictions` table

Run this SQL in the Supabase SQL editor:

```sql
create extension if not exists pgcrypto;

create table if not exists public.predictions (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  age integer,
  sex integer,
  cp integer,
  trestbps integer,
  chol integer,
  fbs integer,
  restecg integer,
  thalach integer,
  exang integer,
  oldpeak double precision,
  slope integer,
  ca integer,
  thal integer,
  threshold double precision,
  probability double precision,
  prediction integer,
  risk_label text
);
```

The app inserts one row per prediction into `public.predictions`. The columns above match the payload built in `src/supabase_client.py`.

---

## 📸 Screenshots

> Add these after running the app

- `reports/figures/app_home.png`
- `reports/figures/app_result.png`

---

## 📂 Project Structure
```bash
heart-disease-risk-stratification/
├── app/
│ └── streamlit_app.py
├── data/
│ ├── raw/
│ └── processed/
├── models/
│ └── heart_model.pkl
├── notebooks/
│ └── 01_eda.ipynb
├── reports/
│ └── figures/
├── src/
│ ├── data_loader.py
│ ├── preprocess.py
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚠️ Limitations

- Small dataset (302 samples)
- Not clinically validated
- No external validation set
- Not suitable for real-world medical use

---

## 📚 Key Learnings

- Data quality > model complexity
- Duplicate leakage can destroy evaluation integrity
- Metrics must align with domain (recall for healthcare)
- Threshold tuning is critical in real systems
- Simpler models can outperform complex ones on structured data

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**.

It is **NOT a medical diagnosis tool** and must not be used for clinical decisions.

---

## 💡 Future Improvements

- Add ROC curve visualization
- Deploy online (Streamlit Cloud)
- Add SHAP explanations
- Collect real-world data
- Build API (FastAPI)

---
