# Revision List: Telco Customer Churn Project

**Target file:** `Telco_Churn_Prediction.ipynb` (21 code cells, 60 cells total)
**Also required:** project deck (PDF), HTML export of the notebook
**Dataset:** `1A_telco_customer_churn.csv` (7,043 rows, 21 columns)

**How cell numbers work:** "Cell N" means the 0-based index of the cell in the notebook (cell 0 is the title cell). Numbers quoted from outputs come from a full top-to-bottom execution of the notebook on this dataset (Restart & Run All succeeded with no errors).

**Priority levels**
- **P1 (blocking):** The notebook makes a claim the output does not support, or a required element is missing. Fix before anything else.
- **P2 (required):** A requirement from the brief is only partially met.
- **P3 (polish):** Clarity, consistency, or reproducibility improvements.

---

## 1. Blocking Issues (P1)

### R1. Final model choice contradicts the validation results
- **Location:** Cell 45 (section 5.5, "Best model" line), Cell 47 (section 6.1 intro), Cell 48 (final model code)
- **Current state:**
  - Cell 42 output: Logistic Regression has the highest validation PR-AUC (0.6543). Tuned Random Forest scores 0.6371 on validation (cell 44).
  - Cell 45 says Random Forest was "selected for strongest PR-AUC on validation set." This is false according to the notebook's own output.
  - Cell 47 says the tuned Random Forest is chosen, with no justification.
- **Required change:** Pick one of two paths and make the text match:
  - **Path A (keep Random Forest):** Write an explicit justification that does not rely on PR-AUC alone. Acceptable reasons include ranking stability across folds, better recall at the top-20% cutoff, or a better fit with the business goal. Support each reason with a number from the notebook. Also evaluate Logistic Regression on the test set (same cell pattern as cell 48) so the comparison is complete.
  - **Path B (switch to Logistic Regression):** Re-run the test evaluation and threshold analysis for Logistic Regression. Update sections 6.2–6.5 to use it. Justify with interpretability (coefficients) and the validation PR-AUC lead.
- **Done when:** Cell 45 and cell 47 state the same choice, the choice is justified with numbers, and the test-set results for both models are available for comparison.

### R2. Feature importance interpretation does not match the output
- **Location:** Cell 55 (section 6.4 interpretation paragraph)
- **Current state:** The paragraph claims `tenure` and `is_new_customer` rank high, and that `num_addon_services` and `has_long_contract` show negative relationships. The actual top 15 from cell 54 is:

  | Rank | Feature | Permutation importance (drop in PR-AUC) |
  |---|---|---|
  | 1 | charge_tenure_ratio | 0.0483 |
  | 2 | Contract | 0.0440 |
  | 3 | tenure | 0.0264 |
  | 4 | has_long_contract | 0.0245 |
  | 5 | InternetService | 0.0230 |
  | 6 | OnlineSecurity | 0.0191 |
  | 7 | TotalCharges | 0.0178 |
  | 8 | TechSupport | 0.0161 |
  | 9 | OnlineBackup | 0.0063 |
  | 10 | PaymentMethod | 0.0043 |
  | 11 | is_new_customer | 0.0040 |
  | 12 | StreamingMovies | 0.0034 |
  | 13 | MultipleLines | 0.0020 |
  | 14 | PhoneService | 0.0013 |
  | 15 | SeniorCitizen | 0.0013 |

  `num_addon_services` does not appear in the top 15.
- **Required change:**
  - Rewrite the paragraph using only the ranking above.
  - Discuss `charge_tenure_ratio` first, since it is the top feature. Explain that it is a engineered feature, and say which of the business expectations it confirmed.
  - Discuss `Contract`, `tenure`, `has_long_contract`, and `InternetService`.
  - Explicitly state that `num_addon_services` and `is_new_customer` ranked lower than expected, and give a reason (for example, `is_new_customer` overlaps with `tenure`, which is already in the model).
  - Compare each feature's observed direction with the expected direction in the feature engineering table (cell 28).
- **Done when:** Every feature named in the paragraph appears in the cell 54 output, and each direction claim is supported by a churn-rate or group comparison in the notebook. If a direction claim is not checked, add a cell that checks it.

### R3. Business recommendations have no numeric impact
- **Location:** Cell 58 (recommendation table), Cell 57 (business impact code)
- **Current state:** The table has three rows. Their "Estimated impact" column says "Net impact per period," "Saved revenue," and "Estimate," with no numbers. Only the whole top-20% program is calculated (cell 57):
  - Customers contacted: 282
  - Churners captured (TP): 185
  - Average MonthlyCharges of churners: 74.44
  - Expected saved revenue: 49,578
  - Program cost: 7,050
  - Net impact: 42,528
  - Lift at top 20%: 1.86x
- **Required change:**
  - Add a calculation cell for each recommendation separately. Suggested approach:
    - **Rec 1 (month-to-month to 1-year migration):** Among top-20% month-to-month customers, count the churners captured. Assume a discount cost and a migration success rate. Calculate net impact.
    - **Rec 2 (onboarding for tenure ≤ 12):** Count captured churners with `is_new_customer == 1` among the 282 contacted. Calculate expected saved revenue and net impact with stated assumptions.
    - **Rec 3 (fiber optic follow-up):** Count captured churners with `InternetService == "Fiber optic"` among the 282 contacted. Calculate impact the same way.
  - Put each number in the table, and list the assumptions under the table.
  - Add a sensitivity note: show how net impact changes if the retention success rate is 15% and 45%.
- **Done when:** Every row in the cell 58 table has a number, a unit, and a referenced assumption. The numbers are reproducible by running a code cell.

### R4. Notebook has no saved outputs
- **Location:** Whole notebook (uploaded version has 0 of 21 code cells with outputs and no execution counts)
- **Required change:** After all other edits, run Restart & Run All and save the notebook so every code cell keeps its output. The brief states that output cells must not be deleted, and the grader needs to see the outputs.
- **Done when:** Every code cell has output, and the saved file opens showing results without re-running.

### R5. Missing deliverables: deck PDF and HTML export
- **Location:** Not in the uploaded files
- **Required change:**
  - Export the notebook to `.html` (File → Download as HTML, or `jupyter nbconvert --to html`).
  - Build the project deck in the outline order (see R13).
- **Done when:** `Telco_Churn_Prediction.html` and `Project_Deck.pdf` exist and every number in the deck matches the notebook output.

---

## 2. Content Gaps Against the Brief (P2)

### R6. EDA insights do not state their numbers
- **Location:** Cells 15, 18, 21, 24 (insight markdown cells after charts 2–5)
- **Current state:**

  | Cell | Chart | Insight text problem | Numbers available in output |
  |---|---|---|---|
  | 15 | Churn by Contract | No numbers stated | 42.7% month-to-month, 11.3% one year, 2.8% two year (cell 14) |
  | 18 | Tenure by churn | "significantly lower," no numbers | Mean tenure 37.6 (no churn) vs 18.0 (churn) (cell 17) |
  | 21 | MonthlyCharges by churn | "more likely," no numbers | Mean 61.27 (no churn) vs 74.44 (churn) (cell 20) |
  | 24 | Churn by InternetService | "notably higher," no numbers | 41.9% fiber, 19.0% DSL, 7.4% no internet (cell 23) |

- **Required change:** Rewrite each insight to quote the numbers above and state the business implication. Remove the word "significantly" unless a statistical test supports it. If you want to use "significant," add a test (for example, a t-test on tenure) and report the p-value.
- **Done when:** Each insight cell contains at least two numbers from the chart's own output.

### R7. Data quality check is not in table form
- **Location:** Cell 8 (section 3.2)
- **Current state:** The check prints values, but there is no table. Outlier checking is missing entirely. The brief lists outliers as a required check.
- **Required change:**
  - Add a markdown table in cell 7 or a new cell after cell 8 with the columns: Aspect, Check performed, Result, Decision.
  - Rows: missing values (0 NaN, 11 blank strings in TotalCharges), data types (TotalCharges read as string), duplicates (0 rows, customerID unique), outliers (new check), class balance (26.5% churn).
  - Add an outlier check using the IQR method on `tenure`, `MonthlyCharges`, and `TotalCharges`. Record the count and the decision (keep, because they are plausible customer values).
- **Done when:** A markdown table exists covering all five checks, and the outlier row has a number and a decision.

### R8. Feature selection leaves a known correlation unaddressed
- **Location:** Cell 33 (section 4.4)
- **Current state:**
  - The code flags `tenure` ↔ `TotalCharges` at r = 0.826, which is above the 0.8 threshold the cell itself uses.
  - It removes `avg_monthly_spend` (r = 0.956 with `MonthlyCharges`).
  - Nothing explains why `tenure`/`TotalCharges` is kept.
- **Required change:** Add a markdown explanation for the `tenure`/`TotalCharges` pair. Either:
  - drop `TotalCharges` (its information is mostly `tenure × MonthlyCharges`), and re-run; or
  - keep both and explain that tree and regularized models tolerate the correlation, with a measured effect on validation PR-AUC.
- **Also:** `avg_monthly_spend` is removed from the preprocessing pipeline only implicitly (ColumnTransformer drops unlisted columns). Make the removal explicit by dropping it from `X` in cell 33, so the feature list printed in cell 29 and the pipeline match.
- **Done when:** Every pair above the threshold has a written decision, and the feature set used by the pipeline is identical to the printed final feature list.

### R9. Comparison table is missing the interpretability column and uses the wrong split
- **Location:** Cell 42 (model comparison code and output), Cell 38 (model description table)
- **Current state:**
  - The table has Model, ROC-AUC, PR-AUC, Precision, Recall, F1, and Train time.
  - The brief requires an interpretability level per model.
  - The table is computed on the validation set. The brief says the comparison should be on the "data uji" (test set). The build guide I wrote also said validation, which contradicts the brief.
- **Required change:**
  - Add an "Interpretability" column (for example: Dummy = none, Logistic Regression = high, coefficients, Random Forest = medium, permutation importance / SHAP).
  - Decide on the split. Option 1: keep validation for model selection, and add a second table with the same models on the test set, labeled as the final comparison. Option 2: compute the table on test only, after the choice in R1 is made. Option 1 is recommended because it keeps the test set single-use. Add a sentence explaining which option was used.
- **Done when:** The table has an interpretability column, and the text states which split the table uses and why.

### R10. Final test results are incomplete for the threshold analysis
- **Location:** Cell 50 (section 6.2)
- **Current state:** Threshold analysis is done for Random Forest only. The confusion matrix at the default 0.5 threshold (cell 48) and the top-20% confusion matrix (cell 50) are both shown but not compared in text.
- **Required change:**
  - Add a short comparison table: default threshold vs top-20% threshold, with precision, recall, F1, and number flagged. Default-threshold values come from cell 48 (classification report and confusion matrix).
  - Explain in one sentence which one is used for the recommendations and why.
- **Done when:** A two-row comparison table exists, and the text says which threshold drives section 6.5.

### R11. Error analysis is thin
- **Location:** Cell 52 (section 6.3)
- **Current state:** The cell prints FN and FP counts and contract mix, and FN vs FP mean tenure and MonthlyCharges. The results are not interpreted in writing. The FP group is 100% month-to-month (cell 52 output), which is an important finding that is not mentioned.
- **Required change:** Add a markdown interpretation after cell 52 covering:
  - FN: 76.2% month-to-month, mean tenure 25.8, mean MonthlyCharges 69.02. What this means for the top-20% cutoff.
  - FP: 100% month-to-month, mean tenure 11.0, mean MonthlyCharges 80.01. What this means for wasted contact budget.
  - One concrete proposal to reduce FN or FP.
- **Done when:** Each of the four group numbers appears in the written interpretation.

### R12. AI usage disclosure lacks the "independent work" section
- **Location:** Cell 59 (final markdown cell)
- **Current state:** The disclosure lists the tool, the parts assisted by AI, and a verification statement. It does not say which decisions were made independently.
- **Required change:** Add a bullet list titled "Independent work" covering at least: choice of primary metric (PR-AUC), threshold rule (top-20% capacity), class imbalance approach, feature selection decisions, interpretation of results, and final recommendation selection. Also remove the line "All business assumptions ... should be validated with the business team before operational use" only if it is not needed. It is fine to keep it as a limitation.
- **Done when:** The disclosure has tool, assisted parts, verification, and independent work sections.

---

## 3. Reproducibility and Clarity (P3)

### R13. Data path does not match the uploaded layout
- **Location:** Cell 6 (`pd.read_csv("data/1A_telco_customer_churn.csv")`) and the header in cell 0
- **Current state:** The code expects the CSV in a `data/` subfolder. The upload places it in the working directory. Running it as-is fails unless the folder is created.
- **Required change:** Either:
  - add a cell at the top that checks the path and falls back to the working directory, for example:
    ```python
    from pathlib import Path
    DATA_PATH = Path("data/1A_telco_customer_churn.csv")
    if not DATA_PATH.exists():
        DATA_PATH = Path("1A_telco_customer_churn.csv")
    df = pd.read_csv(DATA_PATH)
    ```
  - or document the folder layout in cell 0.
- **Done when:** Restart & Run All succeeds with the CSV in either location.

### R14. Business impact unit is unspecified
- **Location:** Cell 57
- **Current state:** The program cost is set to `25` with no unit. Expected saved revenue is in the same unspecified unit.
- **Required change:** Add a comment and a markdown note stating the unit (dataset currency, or a stated conversion). Keep it consistent across R3 and the deck.
- **Done when:** Every monetary figure in the notebook and deck has the same stated unit.

### R15. Build-guide reference in the title cell
- **Location:** Cell 0
- **Current state:** The title cell says "Built per `Instruction_Churn_Project_Build_Guide.md`." That file is not part of the submission.
- **Required change:** Remove the reference, or include the file in the submission folder. Keep the header focused on the project itself.
- **Done when:** Every reference in the notebook points to a file that is included in the submission.

### R16. Make the leakage statement verifiable
- **Location:** Cell 36 (section 4.6)
- **Current state:** The four statements are asserted, not shown.
- **Required change:** Add one short code cell that demonstrates the key claim, for example asserting that `X_test` rows are not present in `X_temp`, and that the fitted preprocessor's scaler means come from training data only. The brief asks for "bukti" (evidence).
- **Done when:** A code cell prints evidence (indices check, or scaler mean compared with train mean) for at least the train-only fitting claim.

---

## 4. Deck (P2)

### R17. Build the project deck in the required outline order
The brief requires this outline, at most 15 main slides plus appendix. The table maps each slide to its notebook source.

| Slide | Content | Notebook source | Revision dependency |
|---|---|---|---|
| 1 | Title, name, student ID, class, dataset code PA-1A | Cell 0 | - |
| 2 | Business problem, decision supported, success metrics | Section 2 | - |
| 3 | Data overview and data quality (7,043 × 21; 11 blank TotalCharges; no duplicates; 26.5% churn) | Cells 6, 8 | R7 |
| 4 | EDA insight 1: target and Contract churn rates | Cells 11, 14 | R6 |
| 5 | EDA insight 2: tenure and MonthlyCharges by churn | Cells 17, 20 | R6 |
| 6 | EDA insight 3: InternetService churn | Cell 23 | R6 |
| 7 | Approach: preprocessing, 5 engineered features, validation scheme | Cells 27–35 | R8 |
| 8 | Model comparison table and tuning before/after | Cells 42, 44 | R9 |
| 9 | Final model: why chosen, test results, confusion matrix, threshold, error analysis | Cells 47–52 | R1, R10, R11 |
| 10 | Feature importance interpretation | Cells 54, 55 | R2 |
| 11 | Recommendations with numeric impact (3+) | Cells 57, 58 | R3 |
| 12 | Limitations and next steps | Cell 59 | - |
| Appendix | AI usage disclosure, technical details | Cell 59 | R12 |

**Deck rules:**
- Slide titles state a conclusion, for example "Month-to-month customers churn at 42.7%, versus 2.8% on two-year contracts."
- One chart per slide.
- Every number must match the notebook. Check each number against the saved HTML export before exporting to PDF.
- Must be understandable without a presenter.

---

## 5. Assumptions to Confirm Before Final Numbers

These are placeholders in the notebook. They must be stated in both the notebook and deck. Confirm or change them before R3 is finalized.

| Assumption | Current value | Where used | Action |
|---|---|---|---|
| Contact capacity | Top 20% (282 of 1,409 test customers) | Cells 50, 57 | Confirm or replace with a real budget figure |
| Retention success probability | 0.30 | Cell 57 | Confirm; add sensitivity at 0.15 and 0.45 (R3) |
| Program cost per contacted customer | 25 (unit unspecified) | Cell 57 | Confirm unit and value (R14) |
| Months of retained revenue | 12 | Cell 57 | Confirm |
| Average revenue per churner | Mean MonthlyCharges of all churners (74.44) | Cell 57 | Confirm; state that it is an all-churner average |

---

## 6. Acceptance Checklist

Mark each item when verified on the final files.

**Notebook**
- [ ] R1: Final model choice is consistent across cells 45, 47, 48, and justified with numbers
- [ ] R2: Feature importance paragraph matches the cell 54 ranking
- [ ] R3: Every recommendation has a number, unit, and assumption; sensitivity included
- [ ] R4: All code cells have saved outputs
- [ ] R6: Insights for cells 15, 18, 21, 24 include numbers
- [ ] R7: Data quality table exists with five checks, including outliers
- [ ] R8: Correlated pairs are addressed; `avg_monthly_spend` removed explicitly
- [ ] R9: Comparison table has interpretability column; split choice is stated
- [ ] R10: Default vs top-20% threshold comparison exists
- [ ] R11: Error analysis interpreted in writing
- [ ] R12: AI disclosure has the independent work section
- [ ] R13: Notebook runs with the CSV in either location
- [ ] R14: Monetary unit is stated everywhere
- [ ] R15: No references to files outside the submission
- [ ] R16: Leakage claim has at least one piece of code evidence
- [ ] Restart & Run All passes with no errors

**Deliverables**
- [ ] R5: HTML export of the final notebook exists
- [ ] R17: Deck PDF follows the 10-point outline plus appendix, at most 15 main slides
- [ ] Every number in the deck matches the notebook
- [ ] Deck appendix contains the AI usage disclosure

**Assumptions**
- [ ] Section 5 assumptions confirmed and stated in notebook and deck

---

## 7. Suggested Order of Work

1. Confirm assumptions in Section 5. Everything in R3 and the deck depends on them.
2. Resolve R1 (model choice). It changes sections 6.1–6.5 and the deck.
3. Apply R8 and R9 (feature set and comparison table). These change the numbers that later sections depend on.
4. Re-run the notebook. Update R2, R3, R6, R10, R11 with the new outputs.
5. Complete R7, R12, R13–R16.
6. Run Restart & Run All, then save with outputs (R4).
7. Export HTML (R5).
8. Build the deck from the final numbers (R17), then check every number against the HTML export.
9. Walk the acceptance checklist.

**Do not change numbers after step 6 without repeating steps 6–8.** Any change to the notebook after the deck is built creates a mismatch.
