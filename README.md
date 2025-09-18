# README: Fraud Detection on Bank Transactions
This repo contains a compact, fraud‑detection workflow built on a (synthetic) bank‑transactions dataset. The project spans from **data cleaning, analysis, & feature engineering** → **model selection and evaluation** with clear figures and a concise write‑up suitable for hiring managers and technical reviewers.

How to read the repo:
- **README (this file):** quick overview + results teaser.  
- **`Reports/REPORT.md`:** deep‑dive narrative (data → features → models → results → error analysis → next steps).  
- **Notebooks:** transparent analysis; kept modular (EDA vs Modeling).  
- **`src/`:** shared helpers to reduce notebook clutter and enhance reproducibility.  
<br>



## Project Overview
This project explores how financial institutions can detect fraudulent activity within large-scale transaction data. Using a synthetic dataset that mimics real-world complexities, I built an end-to-end pipeline from raw records → cleaning → feature engineering → machine learning classification.  

The goal was to identify which model best distinguishes legitimate from fraudulent transactions by spotting unusual patterns in geolocation, transaction descriptions, and device types. Along the way, the project addresses common industry challenges: **class imbalance** (rare fraud cases), **noisy data**, and the trade-off between catching fraud and avoiding false alarms.  

Future work will focus on adding business relevance (e.g., cost-based thresholding and impact analysis) so the model can be evaluated not only technically, but also in terms of operational value. 

## What’s inside (high level)
- **Class imbalance:** handled via `class_weight='balanced'` for baseline models; report **PR–AUC** and optimal thresholding.
- **Timestamps → behavior features:** combined date + time with `datetime` module → `transaction_timestamp`; derived `day_of_week`, `hour`, then **cyclical encodings** (sin/cos)
- **Leakage guards:** encoders/scalers fit **on train only**, then applied to test.  
    - **Categoricals:**
        - Low‑cardinality → **one‑hot** via `OneHotEncoder(drop='first', handle_unknown='ignore')`.
        - High‑cardinality → **frequency encoding** 
        - Domain‑relevant → **target encoding** (fraud rate per category, with global‑rate fallback for unseen).
    - **Numeric scaling:** selected features via `MinMaxScaler`.
- **Evaluation:** cross‑validated model selection (`StratifiedKFold`), ROC curves on test, confusion matrix, and a comparison table.


## Results Preview
- Best model: Random Forest
- ROC–AUC (test): 0.887
- PR–AUC (test): 0.737 
- Engineered features ranked highest in feature importance for all models
- Chosen threshold: F1‑optimal (t = 0.563)
- Confusion matrix @ threshold: TN [36405], FP [0], FN [1197], TP [1998]

<br>
<br>

## Project Structure
```
Portfolio_project1_financialtransactions/
├── Data/
│   ├── raw/                      # Original data (e.g., Bank_Transaction_Fraud1.csv)
│   └── processed/                # Cleaned data artifacts (df_cleaned.csv / .pkl, cv_model_grids2.pkl)
├── Models/                       # Saved trained models (*.pkl)
├── Notebooks/
│   ├── Fraud_DetectionV2.ipynb          # EDA + cleaning + feature engineering → saves processed data
│   └── Fraud_Detection_ModelV2.ipynb    # Feature transformation + Modeling + evaluation → saves models & figures
├── Reports/
│   ├── REPORT.md                 # Full technical report
│   └── figures/*.png             # Exploratory analysis, ROC curves, feature importance, etc.
├── src/
│    └── io_utils.py               # Small I/O helpers (save_fig, save_pkl, etc.)
├── README.md
├── LICENSE.txt
└── .gitignore   (optional but highly recommended)
```

## About the data:
**Dataset:** The dataset used is an *augmented version* of the Kaggle dataset *Bank Transaction Fraud Detection* (by marusagar). Augmentations were added to to mimic real-world fraud patterns → the dataset was adjusted so that certain devices, transaction types, and location inconsistencies are more strongly associated with fraudulent activity while maintaining a realistic overall fraud rate.   
Orginal source link (before augmentation): https://www.kaggle.com/datasets/marusagar/bank-transaction-fraud-detection/data  


## Run workflow  
- `Notebooks/Fraud_DetectionV2.ipynb` → cleans data, engineers features, saves `df_cleaned.*` to `Data/processed/` and figures to `Reports/figures/`.  
- `Notebooks/Fraud_Detection_ModelV2.ipynb` → trains models, tunes hyperparameters (GridSearchCV), evaluates on test set, saves models to `Models/` and figures to `Reports/figures/`.  


## License
This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.





