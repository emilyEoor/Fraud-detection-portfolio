# REPORT — Fraud Detection on Bank Transactions

## 1) Objective
- **Goal:** Detect fraudulent transactions from tabular banking data.  

## 2) Data snapshot
- Original .csv → Rows: 200000; Columns: 22
- Model-ready → Rows: 198000; Columns: 55
- Positive class rate (fraud): 8.07%
- Key fields: transaction_amount, account_balance, transaction_device, transaction_type, transaction_description, transaction_location, city, state, transaction_date, transaction_time, 
- Known issues handled: missing transaction IDs, missing city values, geo mismatches, outliers (IQR‑based detection, not blindly removed).

## 3) Cleaning & feature engineering
Feature engineering aimed to capure realistic fraud signals and handle common data issues:
- **Location quality:** fraud models often learn from missing or inconsistent location fields 
  - `flag_txnloc_missing` for 'Unknown Location'.
  - `flag_geo_mismatch` if `city` or `state` not present in `transaction_location` (recovered some missing cities from location when feasible).
- **Datetime:**
  - Created `transaction_timestamp` from date + time using datetime module → extracted `day_of_week`, `hour`; applied **cyclical encodings** (`dow_sin/dow_cos`, `hour_sin/hour_cos`) to reflect the repeating nature of time; dropped raw.
- **Amount context:** fraud transactions often have unusual scaling relative to account balance.  Computed the following ratio.
  - `txn_to_balance_ratio`, log transformed due to skewed data.
- **Categoricals and Scaling:** All encoders and scalers were fit *only on the training set* and then applied to the test set to prevent data leakage and ensure a realistic evaluation.
    - **Categoricals:**
        - **one‑hot** for low‑cardinality columns (preserves interpretability)
        - **frequency encoding** for high‑cardinality columns (avoids blow-up in features)
        - **target encoding** for `transaction_device`, `transaction_type`, `transaction_description` with mean fraud rate global fallback.
    - **Scaling:** `MinMaxScaler` on selected numerics to normalize ranges for kNN (distance-based) and Logistic Regression (gradient-based) methods.  Tree-based models do not require scaling, but were trained on the same scaled features for consistency across the pipeline.  


## 4) Modeling
- **Data split:** used `train_test_split(test_size=0.2, stratify=y, random_state=42)` → stratify ensures both train and test sets preserve the class imbalance, preventing models from being trained or evaluated on unrepresentative splits.  

- **Model selection:**
    - **Candidates:** Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), kNN - chosen to cover linear, tree-based, and distance-based baselines. 
    - **Hyperparameter tuning:** Applied `GridSearchCV` with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the *training set only*. Stratified folds keep fraud proportions consistent across folds, and CV helps avoid overfitting to a single split. 
    - **Class imbalance:** Set `class_weight='balanced'` for LR/DT/RF so the models weigh fraud cases more heavily, counteracting the rarity of fraud in the dataset.
    - **Statistical comparison of models:** Extracted fold-by-fold ROC–AUC scores at each model’s best parameter setting and applied a Friedman chi-square test across models. This provided a non-parametric way to evaluate whether one model consistently outperformed others across folds, reducing the risk of cherry-picking. KNN excluded prior to test due to substantially lower scores.  

- **Evaluation metrics:**
  - **Primary:** ROC–AUC and Precision‑Recall AUC
    - ROC-AUC measures ranking ability overall, while PR-AUC better reflects the model's performance on rare positives.   
  - **Secondary:** Accuracy, Precision, Recall, F1 
    - for distinguising between closely performing models and interpretability at specific thresholds.
  - **Thresholding:** Initial report metrics at a 0.5 by default, then tuned the classification threshold on the best model (RF) by scanning values between 0 and 1 and selecting the value that maximized the F1 score.  This balances catching fraud (recall), with avoiding false alarms (precision).
- **Model persistence:** Each model was re-fit on the entire training set using the best hyperparameters from cross-validation, then saved to Models/*.pkl ensuring reproducibility and reflecting a deployment ready workflow where trained models can be reloaded for inference.

## 5) Results (test set)

**5.1) Model comparison table (baseline @ 0.50)**

| Model               |   ROC–AUC |   PR–AUC |   Accuracy |   Precision |   Recall |    F1 |
|:--------------------|----------:|---------:|-----------:|------------:|---------:|------:|
| Logistic Regression |     0.888 |    0.737 |      0.895 |       0.413 |    0.706 | 0.521 |
| Decision Tree       |     0.887 |    0.715 |      0.904 |       0.441 |    0.694 | 0.539 |
| Random Forest       |     0.887 |    0.737 |      0.964 |       0.886 |    0.631 | 0.737 |


- All three models show nearly identical ROC–AUC (≈0.887–0.888) and comparable PR–AUC (≈0.715–0.737), indicating similar overall ability to rank fraud vs. legitimate transactions. However, Logistic Regression and Decision Tree achieve higher recall (~0.70) at the cost of very low precision (~0.41–0.44). Random Forest strikes a stronger balance: it has the best overall F1 score (0.737), far higher precision (0.886), and the highest accuracy. This makes Random Forest the most practical choice for fraud detection.

**5.2) Feature importance (top‑10 per model)**  
![Top Features](/Reports/figures/TopFeatures.png)

 - Across all three models, engineered fraud-rate features and simple behavioral flags dominated the top importance rankings (e.g., `transaction_description_fraud_rate`, `transaction_device_fraud_rate`, and `flag_txnloc_missing`). In contrast, raw demographic/location features such as `state` or `city` contributed little predictive power highlighting that domain-informed feature engineering (category-level fraud rates, missing-value flags, ratios) was crucial for separating fraudulent from legitimate transactions in this dataset.

**5.3) ROC & PR Curves for Random Forest Test-Set**  
![ROC & PR Curves](/Reports/figures/roc_pr_curves.png)
-  ROC–AUC shows the model separates fraud from legitimate activity well across thresholds, while PR–AUC highlights that precision–recall trade-offs are more modest.  This reflects the difficulty of catching rare fraud cases without raising false alarms.

**5.4) Threshold tuning impact (F1‑optimal)**

- Operating point chosen via **F1‑optimal** thresholding.  
RF comparison:

| Version                   |   ROC–AUC |   PR–AUC |   Accuracy |   Precision |   Recall |    F1 |   Threshold |
|:---------------------------|----------:|---------:|-----------:|------------:|---------:|------:|------------:|
| Random Forest              |     0.887 |    0.737 |      0.964 |       0.886 |    0.631 | 0.737 |       0.5   |
| Random Forest (F1-optimal) |     0.887 |    0.737 |      0.97  |       1     |    0.625 | 0.769 |       0.563 |

- Instead of defaulting to 0.50, the classification threshold was tuned to maximize the F1 score, resulting in an optimal cutoff at t = 0.563. At this threshold the model achieved a precision of 1, meaning every transaction flagged as fraud in the test set was truly fraudulent. While this looks ideal, it is an artifact of the synthetic dataset (in the real-world, some legitimate transactions inevitably resemble fraud). This cautions that the result reflects dataset limitations rather than deployable performance.  


**5.5) Confusion matrix (@ F1-optimal threshold)**  
![ConfusionMatrix](/Reports/figures/confusion_matrix_best.png)

- The results depicted in the confusion matrix reflect a strong ability to avoid false alarms (precision = 1.0), but at the cost of missing many actual fraud cases.  


## 6) Error analysis (@ F1-optimal threshold)
- **False positives:**
    - The model produced 0 false positives (0.00% of legitimate transactions). While perfect precision is unlikely in real-world deployments, this synthetic dataset produced no false alarms.  

- **False negatives:**
    - The model produced 1,197 false negatives (37.5% of fraud transactions). To better understand missed fraud, I compared false negatives (missed frauds) against true positives (caught frauds) on the model’s top 3 features.  
![False Negative Trends](/Reports/figures/error_analysis_tp_fn_3panel.png)

    - The strongest pattern emerged on `transaction_description_fraud_rate`. Missed fraud clustered in categories with very low historical fraud rates (`transaction_description_fraud_rate ≈ 0`) indicating that fraud cases in historically “safe” categories were harder to distinguish from legitimate activity. 
    - True positives relied heavily on the “missing location” flag (`flag_txnloc_missing`), a signal unlikely to generalize to real-world data.
    - These patterns highlight that the model is strong when fraud aligns with obvious risk signals but struggles with subtle, legitimate-like fraud cases. Real-world improvements would require richer behavioral features (e.g., transaction velocity) and acceptance of some false positives.


## 7) Limitations
- **Synthetic data:**
    - real fraud is **non‑stationary** (concept drift).  
    - real-fraud likely needs temporal cross‑validation; consider **time‑based splits** if timestamps matter.  
    - at the F1 optimal threshold, the RF received a precision of 1 (unrealistic to real-world scenarios)
- **Lack of probability calibration:** 
    - Model outputs are currently used as raw predicted probabilities, but these are not guaranteed to be well-calibrated.  In practice, calibrated probabilities are important for setting business thresholds (e.g., auto-decline vs. manual review).
- **Business Relevance**
    - This project focused on building and evaluating a reproducible fraud-detection pipeline, but it does not yet incorporate business-level constraints such as financial cost of false positives, acceptable fraud loss rates, or operating review thresholds.  In practice, aligning model performance with business impact (e.g., cost-based thresholding, ROI analysis), is critical for deployment.


## 8) Next steps
- Try **CalibratedClassifierCV** for probability calibration.  
- Evaluate **cost‑sensitive** performance once costs are known.  
    -  adjust the classification threshold based on business preference for/against favoring catching more fraud at the expense of more false alarms.
- Package preprocessing + model into an `sklearn.Pipeline` for a single saved artifact and leak‑proof inference.
- Experiment with more advanced imbalance techniques like `BalancedRandomForest`, `XGBoost scale_pos_weight`, or sampling (SMOTE/undersampling) in future experiments.

<br>

---

<br>

---

## Appendix A — Reproduce the exact figures
- Run `Fraud_DetectionV2.ipynb` end-to-end; it produces and saves exploratory plots into `Reports/figures/`, and the cleaned dataset as `df_cleaned.pkl` into `Data/processed/`
- Run `Fraud_Detection_ModelV2.ipynb` end‑to‑end; it writes ROC curves (`model_roc_curves.png`), Precision–Recall curve (`precision_recall_curve.png`), per‑model top features (`TopFeatures.png`), and the confusion matrix at the F1‑optimal threshold (`confusion_matrix_best.png`) into `Reports/figures/`.  It also saves the test models in `Models`.

## Appendix B — Environment
- **OS:** macOS Monterey 12.7.6
- **Python:** python == 3.13.5
- **Core Libraries:**
    - `scikit‑learn` == 1.6.1
    - `pandas` == 2.2.3
    - `numpy` == 2.1.3
    - `matplotlib` == 3.10.0
    - `seaborn` == 0.13.2
    - `scipy` == 1.15.3 

## Appendix C — Data dictionary (excerpt)
| Feature | Type | Description |
|---|---|---|
| `flag_geo_mismatch` | binary | 1 if city/state disagrees with `transaction_location`. |
| `flag_txnloc_missing` | binary | 1 if `transaction_location` is 'Unknown Location'. |
| `*_freq` | numeric | Frequency encoding of high‑cardinality categoricals (fit on train). |
| `*_fraud_rate` | numeric | Target-encoded fraud rate per category (fit on train; global fallback). |
| `hour_sin`, `hour_cos` | numeric | Cyclical encoding of hour of day. |
| `dow_sin`, `dow_cos` | numeric | Cyclical encoding of day of week. |
| `txn_to_balance_ratio` | numeric | Transaction amount divided by account balance (optional log). |

