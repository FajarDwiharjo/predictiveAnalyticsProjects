# Telco Customer Churn Prediction: Build Guide

**Dataset file:** `1A_telco_customer_churn.csv` (uploaded manually to the conversation)
**Dataset details:** 7,043 rows, 21 columns, target column `Churn` (Yes/No, about 26.5% Yes)
**Topic:** Churn prediction (binary classification)

---

## 0. Project Overview

A telecom company loses customers every month and wants to know which customers the retention team should contact first, given a limited budget. The goal is to build a predictive model that ranks customers by churn risk, validate it properly, explain what drives the predictions, and translate the results into concrete business recommendations with estimated impact.

**Deliverables:**
1. A reproducible Jupyter Notebook (`.ipynb`) covering every step below, plus an HTML export
2. A project deck (PDF), maximum 15 main slides plus appendix
3. An AI usage disclosure (final section of this guide and one appendix slide)

**Scope:** Use only the uploaded CSV file. Do not add external data. Deep learning, NLP, computer vision, and deployment are out of scope.

---

## 1. Environment Setup

**Tools:** Python 3.10+, Jupyter Notebook or Google Colab, pandas, numpy, scikit-learn, xgboost or lightgbm, statsmodels, matplotlib, seaborn. Optional: shap, optuna.

```bash
pip install pandas numpy scikit-learn xgboost statsmodels matplotlib seaborn
```

**Notebook header (first cell):**
```python
import warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, confusion_matrix, classification_report,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")
```

**Notebook structure:** Each section starts with a markdown cell explaining the step and the decision taken. Keep all cell outputs. The notebook must run from top to bottom (Restart & Run All) without errors. Set `random_state=42` everywhere randomness is involved.

---

## 2. Business Understanding

Write a one-paragraph problem statement in a markdown cell, covering:

- **Client context:** A telecom company with monthly customer churn.
- **Decision supported:** The retention team has a limited budget and must choose which customers to contact first.
- **Target:** `Churn` (1 = Yes, 0 = No).
- **Model metrics:** ROC-AUC, PR-AUC, precision, recall, F1, and the confusion matrix. Use **PR-AUC** as the primary metric because the classes are imbalanced (about 26.5% positive).
- **Business metrics:** Expected saved revenue and lift in the top-k% of riskiest customers.
- **Justification:** Explain why recall matters (missed churners are lost customers) and why precision also matters (contacting customers who would not churn wastes budget).

---

## 3. Task 1: Data Diagnosis and EDA

### 3.1 Load and inspect
```python
df = pd.read_csv("1A_telco_customer_churn.csv")
print(df.shape)
df.info()
df.head()
```

### 3.2 Data quality table

Check each aspect and summarize findings in a table in the notebook:

| Aspect | How to check | What to look for |
|---|---|---|
| Missing values | `df.isna().sum()` | Also check for blank strings, which pandas does not count as NaN |
| Data types | `df.dtypes` | `TotalCharges` is often read as object instead of numeric |
| Duplicates | `df.duplicated().sum()`, uniqueness of `customerID` | `customerID` should be unique |
| Outliers | Boxplots and IQR on `tenure`, `MonthlyCharges`, `TotalCharges` | Decide whether to keep or cap, and explain |
| Class balance | `df["Churn"].value_counts(normalize=True)` | Record the proportion |

**Known issue to verify:** `TotalCharges` may contain blank strings (usually for customers with `tenure = 0`). Convert with `pd.to_numeric(..., errors="coerce")`, then decide how to handle the resulting NaNs and document the reason (for example, fill with 0 because the customer has no accumulated charges yet, or drop those rows).

```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df["TotalCharges"].isna().sum())
```

### 3.3 Exploratory Data Analysis (at least 5 visualizations)

Each chart must be followed by a markdown cell with a written insight: what the chart shows with actual numbers, and what it implies for the business. Suggested charts:

1. Target distribution (bar chart of `Churn`)
2. Churn rate by `Contract` type
3. `tenure` distribution by churn class (histogram or KDE)
4. `MonthlyCharges` by churn class (boxplot)
5. Churn rate by `InternetService` or `PaymentMethod`
6. Correlation heatmap for numeric features

Example:
```python
churn_by_contract = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean())
churn_by_contract.plot(kind="bar", title="Churn rate by contract type")
plt.ylabel("Churn proportion"); plt.show()
```

Do not write "churn is higher in this group." Write the numbers, for example "Month-to-month customers churn at X%, compared with Y% for two-year contracts," and explain the implication.

---

## 4. Task 2: Data Preparation and Feature Engineering

### 4.1 Basic cleaning
- Drop `customerID` (identifier, not predictive).
- Handle `TotalCharges` as described in Section 3.2.
- Encode the target: `y = (df["Churn"] == "Yes").astype(int)`.
- Treat `SeniorCitizen` (already 0/1) as categorical or binary.

### 4.2 Feature engineering (at least 3 new features, with business logic)

| New feature | Definition | Business rationale | Expected effect on churn |
|---|---|---|---|
| `avg_monthly_spend` | `TotalCharges / (tenure + 1)` | Average bill over the relationship; `+1` avoids division by zero | Depends; test it |
| `num_addon_services` | Count of "Yes" among OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | Customers with more services tend to be stickier | Decreases |
| `is_new_customer` | `1 if tenure <= 12 else 0` | New customers are typically the most vulnerable | Increases |
| `has_long_contract` | 1 if Contract is not month-to-month | Longer contracts raise switching costs | Decreases |
| `charge_tenure_ratio` | `MonthlyCharges / (tenure + 1)` | Captures high bills for relatively new customers | Increases |

For each feature, state the formula, the business reason, and the expected direction of effect. Later, check whether the feature importance results match these expectations.

### 4.3 Preprocessing pipeline
- **Categorical features:** `OneHotEncoder(handle_unknown="ignore")`
- **Numeric features:** median imputation and `StandardScaler`. Scaling is required for logistic regression; tree models do not need it, but using the same pipeline keeps the comparison fair.
- Use a `ColumnTransformer` inside a `Pipeline` so that all fitting happens only on training data.

```python
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend",
            "num_addon_services", "is_new_customer", "has_long_contract", "charge_tenure_ratio"]
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])
```

### 4.4 Feature selection
Apply at least one method and explain it. Options:
- Remove one of two features with very high correlation (for example, `MonthlyCharges` and `avg_monthly_spend`) to reduce multicollinearity.
- Use importance scores from a preliminary tree model to drop features with no predictive value.

List every removed feature and the reason.

### 4.5 Validation scheme

This dataset has no time dimension, so use a stratified split:

```python
X = df.drop(columns=["Churn"])
y = (df["Churn"] == "Yes").astype(int)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE)
# Result: about 60% train, 20% validation, 20% test
```

**Rules for the test set:** Use it only once, in the final evaluation (Section 6). Do not use it for model selection, tuning, or threshold choice. Use `StratifiedKFold` (5 folds) on the training data for tuning, or the validation set.

### 4.6 Leakage check (document in the notebook)
State explicitly that:
- Scaler and imputer are fitted only on training data (guaranteed by the pipeline).
- No feature is derived from the target.
- The test set is not used during tuning.
- No engineered feature uses information from the future.

---

## 5. Task 3: Modeling, Comparison, and Tuning

### 5.1 Three models

| Category | Model | Role |
|---|---|---|
| Baseline | `DummyClassifier(strategy="most_frequent")` | Minimum reference; if other models do not beat it, something is wrong |
| Linear / statistical | `LogisticRegression(class_weight="balanced", max_iter=1000)` | Interpretable coefficients |
| Tree-based ensemble | `RandomForestClassifier` (or XGBoost / LightGBM) | Usually strongest on tabular data |

```python
models = {
    "Baseline (Dummy)": Pipeline([("prep", preprocess),
        ("clf", DummyClassifier(strategy="most_frequent"))]),
    "Logistic Regression": Pipeline([("prep", preprocess),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))]),
    "Random Forest": Pipeline([("prep", preprocess),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=-1))]),
}
```

### 5.2 Class imbalance
Choose one or more approaches and justify the choice:
- `class_weight="balanced"` (used above)
- Resampling such as SMOTE (from `imbalanced-learn`), applied inside each cross-validation fold, not before splitting
- Threshold tuning (covered in Section 6)

### 5.3 Comparison table

Train each model on the training set and evaluate on the validation set:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Training time (s) | Interpretability |
|---|---|---|---|---|---|---|---|
| Baseline | | | | | | | Very high |
| Logistic Regression | | | | | | | High (coefficients) |
| Random Forest | | | | | | | Medium (permutation importance / SHAP) |

Measure training time:
```python
start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start
```

### 5.4 Hyperparameter tuning (at least one model)

Tune Random Forest (or XGBoost) with `GridSearchCV` or `RandomizedSearchCV`, using `StratifiedKFold` and `scoring="average_precision"`:

```python
param_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 8, 14],
    "clf__min_samples_leaf": [1, 5, 10],
}
search = GridSearchCV(models["Random Forest"], param_grid,
                      cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                      scoring="average_precision", n_jobs=-1)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
```

**Report:**
- Score before tuning (default model on validation set) and after tuning (best estimator on validation set)
- Best parameters and tuning time
- Whether the improvement is meaningful. A small improvement is acceptable if you describe it honestly.

### 5.5 Required outputs of this section
- The model comparison table (5.3)
- A before/after tuning table
- One paragraph explaining which model is the most promising and why, with numbers explained, not just listed

---

## 6. Task 4: Final Evaluation, Interpretation, and Recommendations

### 6.1 Final model on the test set

Choose the final model based on PR-AUC, interpretability, and business fit, not on the highest score alone.

Retrain on train + validation data with the best parameters, then evaluate **once** on the test set:

```python
final_model = search.best_estimator_
final_model.fit(X_temp, y_temp)
proba = final_model.predict_proba(X_test)[:, 1]
```

Report ROC-AUC, PR-AUC, precision, recall, F1, and the confusion matrix.

### 6.2 Threshold selection

The threshold must be based on retention program capacity or cost, not a default of 0.5.

Example logic:
- The retention team can contact the **top 20%** of customers (assumption; state it and justify it).
- Of the 1,409 customers in the test set (20% of 7,043), flag the 20% with the highest predicted risk.

```python
k = 0.20
cutoff = np.quantile(proba, 1 - k)
y_pred_topk = (proba >= cutoff).astype(int)

print(confusion_matrix(y_test, y_pred_topk))
print("Precision:", precision_score(y_test, y_pred_topk))
print("Recall:", recall_score(y_test, y_pred_topk))
```

Alternative: derive the threshold from costs. If contacting a customer costs X and a successfully retained customer is worth Y in expected value, contact a customer when `probability × Y − X > 0`. Choose one approach and state the assumptions.

### 6.3 Error analysis

- **False negatives (churners missed):** What characteristics do they share? Compare their feature distributions with correctly identified churners.
- **False positives (contacted but did not churn):** What is the cost impact?

```python
errors = X_test.copy()
errors["y_true"] = y_test.values
errors["y_pred"] = y_pred_topk
fn = errors[(errors.y_true == 1) & (errors.y_pred == 0)]
fp = errors[(errors.y_true == 0) & (errors.y_pred == 1)]
print(len(fn), len(fp))
print(fn["Contract"].value_counts(normalize=True))
```

### 6.4 Feature importance and interpretation

Use permutation importance on the test set (or SHAP for tree models):

```python
perm = permutation_importance(final_model, X_test, y_test, n_repeats=10,
                              scoring="average_precision", random_state=RANDOM_STATE)
imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
imp.head(15).plot(kind="barh", title="Permutation importance (drop in PR-AUC)")
plt.gca().invert_yaxis(); plt.show()
```

Write an interpretation, not just a chart. Example: "`Contract` is the most influential feature. Shuffling it reduces PR-AUC by X, which indicates..." Check whether the direction of each effect matches your expectations from Section 4.2, and explain any mismatch.

### 6.5 Business impact (at least 3 recommendations)

Calculate expected saved revenue or lift for the top 20%.

**Calculation framework (all assumptions must be stated and reasonable):**

| Variable | Value | Source / reason |
|---|---|---|
| Test customers | 1,409 | 20% of 7,043 |
| Customers contacted (top 20%) | 282 | Capacity assumption |
| Churners captured (TP) | from confusion matrix | Model result |
| Average `MonthlyCharges` of churners | from data | Compute from dataset |
| Probability retention succeeds | e.g., 30% | Assumption; justify |
| Program cost per contacted customer | e.g., X | Assumption |

```
Expected saved revenue = TP_contacted × P(retention success) × avg_MonthlyCharges × months (e.g., 12)
Net impact = Expected saved revenue − (customers_contacted × program_cost)
Lift at top 20% = Recall_top20 / population churn rate (≈ 0.265)
```

Use one consistent currency. The dataset does not specify one, so write "dataset monetary units" or state a conversion assumption clearly.

Build a recommendation table with at least 3 rows:

| # | Finding (from data) | Recommended action | Estimated impact | Priority |
|---|---|---|---|---|
| 1 | Month-to-month churn is much higher | Offer discounted migration to 1-year contracts to top-20% month-to-month customers | Net impact X per period | High |
| 2 | New customers (tenure ≤ 12 months) are vulnerable | 90-day onboarding and check-in program | Saved revenue Y | High |
| 3 | Customers without online security services are at higher risk | Bundle security or tech support services with a promotion | Estimate Z | Medium |

List the assumptions below each estimate.

---

## 7. Project Deck (PDF)

Maximum 15 main slides plus appendix. Design is free, but follow this order:

| Slide | Content | Source in notebook |
|---|---|---|
| 1 | Title, dataset name (Telco Customer Churn), author, date | - |
| 2 | Business problem and objective: client, decision supported, success metrics | Section 2 |
| 3 | Data overview and quality: size (7,043 × 21), issues found (TotalCharges, etc.), how they were handled | Section 3.2 |
| 4–6 | Key EDA insights (maximum 3 slides, one main message each) | Section 3.3 |
| 7 | Approach: preprocessing, feature engineering (3+ features), validation scheme | Section 4 |
| 8 | Model comparison table and before/after tuning results | Section 5 |
| 9 | Final model: why chosen, test results, confusion matrix, threshold, error analysis | Section 6.1–6.3 |
| 10 | Interpretation: most influential factors | Section 6.4 |
| 11 | Business recommendations and estimated impact (3+) | Section 6.5 |
| 12 | Limitations and next steps | Reflection |
| Appendix | AI usage disclosure (see Section 9 below), technical details | - |

**Deck principles:**
- Slide titles state the conclusion, for example "Month-to-month customers churn three times more often," not "Churn by contract."
- One chart per slide, with clear labels.
- The deck must be understandable without a presenter.
- Every number must match the notebook. Re-check before exporting to PDF.

---

## 8. Final Quality Checklist

**Notebook**
- [ ] Restart & Run All runs without errors
- [ ] `random_state=42` used in models, splits, and cross-validation
- [ ] Every section has a markdown explanation
- [ ] Cell outputs are kept
- [ ] HTML export completed
- [ ] Test set used only once, in final evaluation

**Data and modeling**
- [ ] Data quality table present
- [ ] At least 5 EDA charts, each with written insight
- [ ] `TotalCharges` handled with a documented reason
- [ ] At least 3 new features with business rationale
- [ ] Preprocessing inside a pipeline, no leakage
- [ ] At least 3 models (baseline, linear, tree ensemble)
- [ ] Comparison table with ROC-AUC, PR-AUC, precision, recall, F1, and training time
- [ ] Hyperparameter tuning with before/after scores
- [ ] Class imbalance handled and justified

**Churn-specific**
- [ ] ROC-AUC, PR-AUC, precision, recall, F1, and confusion matrix reported
- [ ] Threshold chosen from capacity or cost, not 0.5 by default
- [ ] Business impact calculated (expected saved revenue or top-k% lift)

**Interpretation and recommendations**
- [ ] Permutation importance or SHAP interpreted
- [ ] False negative and false positive analysis done
- [ ] At least 3 recommendations with estimated impact and stated assumptions

**Deck**
- [ ] Maximum 15 main slides plus appendix
- [ ] Slide order matches Section 7
- [ ] Conclusion-style slide titles
- [ ] All numbers match the notebook
- [ ] Exported to PDF

---
