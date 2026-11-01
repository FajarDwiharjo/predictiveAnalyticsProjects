# Telco Customer Churn Prediction

Binary classification project that ranks telecom customers by churn risk, validates the model properly, explains what drives the predictions, and turns the results into concrete retention recommendations with estimated financial impact.

**Author:** NAvrosjo (`navrosjo@gmail.com`) — see pyproject.toml  
**Python:** 3.12 (uv project)  
**Dataset:** `data/1A_telco_customer_churn.csv` — 7,043 customers × 21 columns  
**Primary metric:** PR-AUC (churn is imbalanced: ~26.5% positive)

---

## Project description

A telecom company loses customers every month. The retention team has a limited budget and must decide which customers to contact first. This project builds a predictive model that:

1. Ranks customers by predicted churn probability.
2. Is validated on a held-out test set used exactly once (no leakage, no tuning on test).
3. Explains which features drive churn and in which direction.
4. Translates predictions into business recommendations with estimated saved revenue, program cost, and net impact.

The work follows `Instruction_Churn_Project_Build_Guide.md` and is tracked issue-by-issue in `revision_list.md` (Round 2).

---

## Deliverables

| File | Purpose |
|------|---------|
| `Telco_Churn_Prediction.ipynb` | Full Jupyter notebook, all 62 cells executed with saved outputs |
| `Telco_Churn_Prediction.html` | HTML export of the notebook (for review without running code) |
| `Project_Deck.pdf` | Project presentation deck (max 15 main slides + appendix) |
| `make_notebook.py` / `build_deck.py` | Scripts that generate the notebook and deck |
| `verify_notebook.py` | Checks that every code cell has saved output |
| `revision_list.md` | Round 2 review notes and acceptance checklist |
| `Instruction_Churn_Project_Build_Guide.md` | Original build guide / specification |
| `README.md` | This file |

---

## Dataset explanation

**File:** `data/1A_telco_customer_churn.csv`  
**Shape:** 7,043 rows × 21 columns  
**Target:** `Churn` (Yes/No) — 1,869 churned (26.5%), 5,174 did not (73.5%)

| Column | Type | Notes |
|--------|------|-------|
| `customerID` | string | Identifier, dropped before modeling |
| `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` | categorical | One-hot encoded in the pipeline |
| `SeniorCitizen` | int (0/1) | Treated as binary |
| `tenure` | int | Months as a customer |
| `MonthlyCharges` | float | Monthly bill |
| `TotalCharges` | string→numeric | Read as text; 11 blank strings (all `tenure = 0`), coerced to numeric and filled with 0 |
| `Churn` | string | Target (Yes → 1, No → 0) |

**Data quality findings:**
- Missing values (NaN): 0
- Blank strings in `TotalCharges`: 11 (all customers with `tenure = 0`, filled with 0)
- Duplicate rows: 0; `customerID` is unique
- Outliers (IQR method on `tenure`, `MonthlyCharges`, `TotalCharges`): 0 in all three columns
- Class balance: 26.5% churn / 73.5% no churn

**Engineered features (5 created, 4 kept after selection):**
| Feature | Definition | Business rationale | Expected effect |
|---------|------------|--------------------|-----------------|
| `avg_monthly_spend` | `TotalCharges / (tenure + 1)` | Average bill over relationship | Tested; removed (r = 0.956 with `MonthlyCharges`) |
| `num_addon_services` | Count of "Yes" among 6 add-on services | More services = stickier customer | Decreases churn |
| `is_new_customer` | `1 if tenure <= 12 else 0` | New customers are most vulnerable | Increases churn |
| `has_long_contract` | 1 if Contract is not month-to-month | Longer contracts raise switching cost | Decreases churn |
| `charge_tenure_ratio` | `MonthlyCharges / (tenure + 1)` | High bill for a relatively new customer | Increases churn |

**Validation scheme (stratified, no time dimension):** ~60% train / 20% validation / 20% test. The test set (1,409 customers) is used exactly once, in final evaluation.

**Leakage check (documented in notebook cell 36):** scaler/imputer fitted on training data only; no feature derived from the target; test rows absent from training (overlap = 0); tuning uses `StratifiedKFold` on training data only.

---

## Project guide

The full specification is in `Instruction_Churn_Project_Build_Guide.md`. The notebook follows its section order:

- **Section 2 — Business Understanding:** client context, decision supported, target, metrics (ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix), business metrics (saved revenue, lift), metric justification.
- **Section 3 — Data Diagnosis and EDA:** load/inspect, data-quality table, 5 charts each with a numbered insight.
- **Section 4 — Data Preparation and Feature Engineering:** cleaning, 5 engineered features, preprocessing pipeline (`ColumnTransformer` + `Pipeline`), feature selection (drop `avg_monthly_spend`), validation scheme, leakage evidence.
- **Section 5 — Modeling, Comparison, and Tuning:** 3 models (Dummy baseline, Logistic Regression, Random Forest), class imbalance (`class_weight="balanced"`), validation comparison table, Random Forest hyperparameter tuning via `GridSearchCV` (PR-AUC scoring).
- **Section 6 — Final Evaluation, Interpretation, and Recommendations:** final model on test set, threshold selection (top-20% capacity), error analysis (FN/FP), permutation importance, business impact with 3 recommendations.

---

## Key results (from the executed notebook)

**Model comparison on validation set:**

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Interpretability |
|-------|---------|--------|-----------|--------|----|------------------|
| Baseline (Dummy) | 0.5000 | 0.2654 | 0.0000 | 0.0000 | 0.0000 | None |
| Logistic Regression | 0.8402 | 0.6543 | 0.5223 | 0.7834 | 0.6267 | High (coefficients) |
| Random Forest | 0.8177 | 0.5983 | 0.5569 | 0.6283 | 0.5905 | Medium (permutation importance / SHAP) |

**Tuning (Random Forest):** best params `max_depth=None, min_samples_leaf=10, n_estimators=400`; best CV PR-AUC 0.6768; default PR-AUC 0.5983 → tuned 0.6405 (+0.0423); tuning time 45.03 s.

**Final model:** Logistic Regression (chosen on validation PR-AUC 0.6543, retrained on train + validation, evaluated once on the test set).

**Logistic Regression — test set:**
- ROC-AUC: 0.8464 | PR-AUC: 0.6644
- Confusion matrix: `[[751, 284], [85, 289]]` (TN, FP, FN, TP)
- Churn class: precision 0.50, recall 0.77, F1 0.61

**Threshold selection (top-20% capacity, cutoff 0.7383):**
- 282 of 1,409 test customers flagged
- Precision 0.6596 | Recall 0.4973 | F1 0.5671 | TP 186 | FN 188 | FP 96
- Lift at top-20%: 1.87× (population churn rate 0.265)
- Used for recommendations because it matches retention-team contact capacity

**Error analysis (top-20% threshold):**
- False negatives (188 missed churners): 76.6% month-to-month, mean tenure 25.2, mean charges 67.65
- False positives (96 contacted, did not churn): 99.0% month-to-month, mean tenure 11.9, mean charges 80.35

**Feature importance (permutation, drop in PR-AUC):**
1. `tenure` 0.1042
2. `MonthlyCharges` 0.0899
3. `InternetService` 0.0755
4. `charge_tenure_ratio` 0.0741
5. `has_long_contract` 0.0364
6. `Contract` 0.0159

**Business impact (top-20% retention program):**
- Expected saved revenue: 49,846 | Program cost: 7,050 | Net impact: 42,796 | Lift: 1.87×
- Retention success probability: 0.30 | Program cost per customer: 25 (dataset monetary units)
- Sensitivity: net impact 17,873 at 15% retention → 67,719 at 45% retention

**Recommendations:**

| # | Finding | Action | Estimated impact | Priority |
|---|---------|--------|------------------|----------|
| 1 | Month-to-month churn 42.7% vs 2.8% two-year | Discounted migration to 1-year for flagged month-to-month customers | 61,904 net impact / period | High |
| 2 | New customers (tenure ≤ 12) vulnerable | 90-day onboarding for flagged new customers | 41,640 net impact / period | High |
| 3 | Fiber optic churn 41.9% vs 7.4% no-internet | Service-quality follow-up for flagged fiber customers | 28,309 net impact / period | Medium |

Assumptions are listed under the table in the notebook (cell 60).

---

## How to run

### Environment setup

```bash
# Clone / open this folder, then:
uv sync
```

Requires Python 3.12+. Dependencies (from `pyproject.toml`): pandas, numpy, scikit-learn, xgboost, statsmodels, matplotlib, seaborn, jupyter, shap, optuna.

### Run the notebook

```bash
uv run jupyter notebook Telco_Churn_Prediction.ipynb
```

Then: **Kernel → Restart & Run All**. All 62 cells should execute without error and every code cell should retain its output.

Verify saved outputs:
```bash
uv run python verify_notebook.py
```

### Export HTML

```bash
uv run jupyter nbconvert --to html Telco_Churn_Prediction.ipynb
```

### Regenerate the deck

```bash
uv run python build_deck.py
```

The deck must be rebuilt from the final notebook outputs; do not retype numbers by hand.

### Notes

- The notebook reads `data/1A_telco_customer_churn.csv` from the project root (cell 1 resolves `DATA_PATH`). The file is committed to the repo under `data/`.
- `random_state = 42` is used everywhere randomness is involved.
- One known output issue: cell 1 currently prints an absolute local path (`C:\Users\NAvroshy\...`). This is flagged as item A8 in `revision_list.md` and should be removed before any external sharing.

---

## Project status

The notebook is executed and saved with outputs, and the HTML export and deck exist. Remaining review work is tracked in **`revision_list.md`** (Round 2), which is the authoritative source for what still needs fixing before final submission. Key open items include:

- **Notebook (Part A):** A1 feature-importance text order, A2 sensitivity figures in cell 60, A3 written error analysis, A4 meaningless calc in cell 52, A5 outlier wording, A6 Rec 2 mislabeling, A7 direction claims unverified, A8 local path in output.
- **Deck (Part B):** B1–B12 — dataset dimensions (7,043 × 21, not 26), metric values, interpretability column, tuning params, confusion matrix, threshold metrics, feature chart, recommendation numbers, engineered-feature count, NIM/name on title slide, outlier wording, slide–notebook consistency.
- **Repo (Part C):** run Restart & Run All, re-export HTML, regenerate deck, cross-check every deck number against the HTML export, fill in this README (done), commit final files.

See `revision_list.md` for the full acceptance checklist.

---

## AI usage disclosure

AI tooling was used to draft parts of the notebook, deck scripts, and this README. All modeling decisions (primary metric PR-AUC, top-20% threshold rule, class-imbalance approach, feature-selection decisions, result interpretation, final recommendation selection) were made independently. Business assumptions (contact capacity 20%, retention success 30%, program cost 25 per customer, 12 months of revenue) are estimates stated in the notebook and should be validated with the business team before operational use. Full disclosure is in the notebook's final cell and the deck appendix.

---

*Generated from the executed notebook `Telco_Churn_Prediction.ipynb` and `Instruction_Churn_Project_Build_guide.md`. Numbers quoted here come from saved cell outputs; regenerate via Restart & Run All if the data or code changes.*
