# Revision List (Round 2): Telco Churn Project

**Files reviewed:** `Telco_Churn_Prediction_1.ipynb` (saved outputs, not re-run) and `Project_Deck.pdf`
**Dataset:** `1A_telco_customer_churn.csv` (7,043 rows, 21 columns)

**Status:** The notebook meets most of Round 1. Items below are what's left. The deck was built from an earlier run and must be regenerated after the notebook is fixed.

**Cell numbers** are 0-based indices in `Telco_Churn_Prediction_1.ipynb`.

**Priority**
- **P1:** Wrong or contradictory claim, or a bug that changes a reported number.
- **P2:** Requirement only partly met.
- **P3:** Clarity or presentation.

---

## Part A: Notebook

### A1. Feature importance interpretation contradicts the output (P1)
- **Location:** Cell 56 (markdown, section 6.4)
- **Problem:** Cell 55 output ranks features as:
  1. tenure 0.1042
  2. MonthlyCharges 0.0899
  3. InternetService 0.0755
  4. charge_tenure_ratio 0.0741
  5. has_long_contract 0.0364
  6. Contract 0.0159

  The text instead says `charge_tenure_ratio` is most influential, `Contract` is second, `tenure` is third, `has_long_contract` is fourth, and `InternetService` is fifth. It also claims "the ranking above is used exactly as produced." That is false.
- **Required change:** Rewrite the paragraph in the order of the cell 55 output. Name each feature with its actual rank and value. Remove the claim that the text matches the output unless it does.
- **Done when:** Every rank and value in the paragraph matches the cell 55 output.

### A2. Sensitivity figures in the recommendation table are wrong (P1)
- **Location:** Cell 60
- **Problem:** The sensitivity line is computed as `net_impact * 0.15 / 0.30` and `net_impact * 0.45 / 0.30`. This scales net impact and ignores the fixed program cost. It gives 21,398 to 64,194.
- **Correct values** (from cell 58, which uses the right formula): 17,873 at 15% and 67,719 at 45%.
- **Required change:** Recompute sensitivity in cell 60 with `tp_contacted × p × avg_monthly_charges_churners × 12 − program_cost`, or reuse the values from cell 58.
- **Done when:** Cell 60 and cell 58 print the same sensitivity values.

### A3. Error analysis has no written interpretation (P2, R11)
- **Location:** After cell 53
- **Problem:** Cell 53 prints the FN and FP profiles, but no markdown cell explains them.
- **Required change:** Add a markdown cell after cell 53 covering:
  - FN (188): 76.6% month-to-month, mean tenure 25.2, mean MonthlyCharges 67.65. What this means for the top-20% cutoff.
  - FP (96): 99.0% month-to-month, mean tenure 11.9, mean MonthlyCharges 80.35. What this means for wasted contact budget.
  - One concrete change that would reduce FN or FP.
- **Done when:** Each of the eight numbers above appears in the written interpretation.

### A4. Meaningless calculation in cell 52 (P1)
- **Location:** Cell 52, last print block
- **Problem:** "FN churn rate by InternetService" is `groupby(...)["y_true"].mean()` on the FN set. Every FN has `y_true = 1`, so every group shows 1.0. The output carries no information.
- **Required change:** Replace it with the share of FN by InternetService, for example `fn["InternetService"].value_counts(normalize=True)`. Make the label say what it shows.
- **Done when:** The printed output varies across groups and the label describes it correctly.

### A5. Outlier text contradicts the code (P1)
- **Location:** Cell 7 (data quality table, "Outliers" row)
- **Problem:** The table says "Outliers exist but are plausible customer values." Cell 6 found 0 outliers for `tenure`, `MonthlyCharges`, and `TotalCharges` under the IQR rule.
- **Required change:** Change the row to state the count (0 for all three columns) and the decision (no action needed). Keep the note that the rule is IQR-based.
- **Done when:** Cell 7 and cell 6 agree.

### A6. Rec 2 finding is mislabeled (P2)
- **Location:** Cell 60, recommendation table, row 2
- **Problem:** The finding reads "New customers (tenure <= 12) mean tenure 18.0 vs 37.6." Those values compare churners with non-churners, not new customers with other customers.
- **Required change:** Rewrite the finding to describe what the data shows about new customers. Either use the churn rate for `is_new_customer == 1` against `is_new_customer == 0`, or cite the 18.0 vs 37.6 figures and label them as the mean tenure of churners vs non-churners.
- **Done when:** The finding wording and the numbers describe the same comparison.

### A7. Direction claims are asserted, not checked (P2)
- **Location:** Cell 56, last paragraph
- **Problem:** The text says each feature's direction matches the business expectation. No code verifies the direction.
- **Required change:** Add a short code cell that prints the Logistic Regression coefficients for `Contract`, `tenure`, `has_long_contract`, `is_new_customer`, `charge_tenure_ratio`, and `num_addon_services`. Use the signs to confirm or correct each direction claim. Scaled features make the sign meaningful, since `StandardScaler` is in the pipeline.
- **Done when:** Each direction claim in cell 56 is backed by a coefficient sign printed in the notebook.

### A8. Personal path in output (P3)
- **Location:** Cell 1 output
- **Problem:** The output prints `C:\Users\NAvroshy\Documents\GitHub\...`.
- **Required change:** Replace the printed path with the relative path, or remove the print statement.
- **Done when:** No local machine path appears in any saved output.

---

## Part B: Deck (`Project_Deck.pdf`)

The deck must be regenerated after Part A is done. Use the numbers below as the check list. Every number must match the final notebook output.

### B1. Dataset dimensions are wrong (P1)
- **Slides:** 1 (title) and 3 (data overview)
- **Problem:** The deck says 7043 × 26 columns. The dataset has 21 columns.
- **Required change:** Change to 7,043 × 21 everywhere.

### B2. Model comparison table does not match the notebook (P1)
- **Slide:** 8
- **Problem and correct values (cell 42, validation set):**

  | Model | Deck shows | Notebook shows |
  |---|---|---|
  | Logistic Regression ROC-AUC | 0.840 | 0.8402 (OK) |
  | Logistic Regression PR-AUC | 0.654 | 0.6543 (OK) |
  | Logistic Regression precision | 0.541 | 0.5223 |
  | Logistic Regression recall | 0.735 | 0.7834 |
  | Logistic Regression F1 | 0.624 | 0.6267 |
  | Random Forest ROC-AUC | 0.826 | 0.8177 |
  | Random Forest PR-AUC | 0.618 | 0.5983 |
  | Random Forest precision | 0.566 | 0.5569 |
  | Random Forest recall | 0.666 | 0.6283 |
  | Random Forest F1 | 0.612 | 0.5905 |

- **Required change:** Copy the values from the cell 42 output table. Use the same rounding as the notebook.

### B3. Interpretability column is empty (P1)
- **Slide:** 8
- **Problem:** The column shows "-" for every row. Cell 42 has real values: None, High (coefficients), Medium (permutation importance / SHAP).
- **Required change:** Fill the column from cell 42.

### B4. Tuning results do not match (P1)
- **Slide:** 8
- **Problem and correct values (cell 44):**
  - Deck says default PR-AUC 0.6175, tuned 0.6348, improvement +0.0173.
  - Notebook says default 0.5983, tuned 0.6405, improvement +0.0423.
  - Deck says best params `max_depth=14, min_samples_leaf=1, n_estimators=400`.
  - Notebook says best params `max_depth=None, min_samples_leaf=10, n_estimators=400`. Best CV PR-AUC 0.6768.
- **Required change:** Replace with the cell 44 values. Show the best CV score and tuning time if space allows.

### B5. Final model test results do not match (P1)
- **Slide:** 9
- **Problem and correct values (cell 48):**
  - LR test ROC-AUC 0.8464 (deck 0.846, OK). PR-AUC 0.6644 (deck says 0.663).
  - Confusion matrix is [[751, 284], [85, 289]]. The deck says TN 796, FP 239, FN 92, TP 282. The deck's TP 282 equals the number flagged at top-20%, not the true positives in the matrix.
  - Random Forest test ROC-AUC 0.8445, PR-AUC 0.6542. The deck says 0.836 and 0.627.
- **Required change:** Replace the confusion matrix and the RF test figures with the cell 48 values.

### B6. Threshold metrics do not match (P1)
- **Slide:** 9
- **Problem and correct values (cell 50, top-20%):** Precision 0.6596 (deck 0.667), recall 0.4973 (deck 0.503), lift 1.87× (deck 1.89×). TP is 186.
- **Required change:** Use the cell 50 and cell 58 values.

### B7. Feature importance chart and text come from a different run (P1)
- **Slide:** 10
- **Problem:** The bar chart and the text list values that do not appear in the final notebook output:
  - Deck: charge_tenure_ratio 0.0737, tenure 0.0736, MonthlyCharges 0.0554, InternetService 0.0550.
  - Notebook (cell 55): tenure 0.1042, MonthlyCharges 0.0899, InternetService 0.0755, charge_tenure_ratio 0.0741.
  - Deck chart shows PaperlessBilling in the top 10. In the notebook it ranks 12th.
  - Deck text says charge_tenure_ratio is the top predictor. In the notebook it ranks 4th.
- **Required change:** Redraw the chart from the cell 55 values, top 10 or top 15. Rewrite the bullets and the title to match. The title "charge_tenure_ratio and Contract are the top predictors" must change, since `tenure` is the top predictor and `Contract` ranks 6th.

### B8. Recommendation numbers do not match (P1)
- **Slide:** 11
- **Problem and correct values (cell 60 and cell 58):**

  | Item | Deck shows | Notebook shows |
  |---|---|---|
  | Rec 1 net impact | 62,619 | 61,904 |
  | Rec 1 churners captured | 187 of 280 | 185 of 280 |
  | Rec 2 net impact | 41,660 | 41,640 |
  | Rec 2 churners captured | 147 of 215 | 147 of 216 |
  | Rec 3 net impact | 28,918 | 28,309 |
  | Rec 3 churners captured | 162 of 242 | 159 of 240 |
  | Whole program net impact | 43,332 | 42,796 |
  | Whole program churners | 188 | 186 |
  | Whole program lift | 1.89× | 1.87× |
  | Sensitivity (15% to 45%) | 21,666 to 64,998 | 17,873 to 67,719 (after A2) |

- **Required change:** Use the final values after A2 is fixed.

### B9. Engineered feature count is wrong (P2)
- **Slide:** 4 (approach)
- **Problem:** The deck says "Engineered features (5)" and then lists `avg_monthly_spend` as removed. Four features are kept.
- **Required change:** Say "5 engineered, 4 kept. `avg_monthly_spend` removed (r = 0.956 with MonthlyCharges)."

### B10. Title slide is missing NIM and the author name is unclear (P1)
- **Slide:** 1
- **Problem:** The brief requires name, NIM, class, and dataset code on the title slide. The deck has no NIM. It says "Prepared by: NAvrosjo". The notebook path on your machine reads `NAvroshy`.
- **Required change:** Add your NIM. Confirm your name spelling and use it consistently on the title slide and the appendix.

### B11. Outlier claim in the deck (P3)
- **Slide:** 3
- **Problem:** The deck says outliers were checked and "kept as plausible values." The notebook found 0 outliers.
- **Required change:** Say "IQR check: 0 outliers in tenure, MonthlyCharges, TotalCharges."

### B12. Interpretability and business slides must match the notebook after the fixes
- **Slides:** 10, 11
- **Required change:** After A1, A2, and A7 are done, check every sentence on slides 10 and 11 against the notebook text.

---

## Part C: Final steps (after Part A and Part B)

1. Run Restart & Run All in `Telco_Churn_Prediction_1.ipynb`. Confirm it finishes with no errors.
2. Save the notebook with outputs.
3. Export the HTML again (`Telco_Churn_Prediction.html`).
4. Regenerate the deck from the saved outputs. Do not type numbers from memory.
5. Check every number in the deck against the HTML export. Use this list to tick each item:
   - [ ] Dataset 7,043 × 21
   - [ ] LR and RF validation table (B2)
   - [ ] Interpretability column filled (B3)
   - [ ] Tuning before and after (B4)
   - [ ] LR test PR-AUC, confusion matrix, RF test (B5)
   - [ ] Threshold precision, recall, lift (B6)
   - [ ] Feature importance chart and bullets (B7)
   - [ ] Recommendation table and sensitivity (B8)
   - [ ] Engineered feature count (B9)
   - [ ] NIM and name on title slide (B10)
   - [ ] Outlier wording (B11)
6. Check the AI usage disclosure appendix still matches the work done.
7. Commit the final files to the repo and update the README (still 0 bytes in the last review).

---

## Acceptance checklist

**Notebook**
- [ ] A1 feature importance text matches cell 55 output
- [ ] A2 sensitivity values consistent between cells 58 and 60
- [ ] A3 written error analysis after cell 53
- [ ] A4 cell 52 output varies and is labeled correctly
- [ ] A5 outlier wording matches cell 6
- [ ] A6 rec 2 finding describes the correct comparison
- [ ] A7 direction claims backed by coefficient signs
- [ ] A8 no personal local path in outputs

**Deck**
- [ ] B1 to B11 all checked against the final notebook
- [ ] B12 slides 10 and 11 text matches the notebook

**Repo**
- [ ] Final notebook, HTML, and deck committed
- [ ] README filled in