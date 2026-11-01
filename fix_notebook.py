import json
from pathlib import Path

path = Path("Telco_Churn_Prediction.ipynb")
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ---------- A8: Cell 1 - remove personal absolute path ----------
c1 = cells[1]
src = "".join(c1["source"])
src = src.replace(
    'print(f"Reading dataset from: {DATA_PATH.resolve()}")',
    'print(f"Reading dataset from: {DATA_PATH}")'
)
c1["source"] = src.splitlines(keepends=True)

# ---------- A5: Cell 7 - fix outlier text to match code (0 outliers) ----------
c7 = cells[7]
src = "".join(c7["source"])
src = src.replace(
    "| Outliers | IQR method on `tenure`, `MonthlyCharges`, `TotalCharges` | Outliers exist but are plausible customer values | Keep (no capping); document as legitimate variation |",
    "| Outliers | IQR method on `tenure`, `MonthlyCharges`, `TotalCharges` | 0 outliers found in all three columns | No action needed; values within plausible range |"
)
c7["source"] = src.splitlines(keepends=True)

# ---------- A4: Cell 52 - fix meaningless FN churn rate calc ----------
c52 = cells[52]
src = "".join(c52["source"])
src = src.replace(
    'print("FN churn rate by InternetService:")\nprint(fn.groupby("InternetService")["y_true"].mean().round(3))',
    'print("FN share by InternetService:")\nprint(fn["InternetService"].value_counts(normalize=True).round(3))'
)
c52["source"] = src.splitlines(keepends=True)

# ---------- A1: Cell 56 - rewrite interpretation to match cell 55 order ----------
new56 = """**Interpretation (R2).** The ranking below follows the cell 55 output exactly.

- **`tenure`** (1st, 0.1042) is the most influential feature. Longer tenure means lower churn risk, consistent with Insight 3 and the expected negative direction.
- **`MonthlyCharges`** (2nd, 0.0899) is second. Higher monthly charges increase churn risk, consistent with Insight 4.
- **`InternetService`** (3rd, 0.0755) is third. Fiber optic customers churn most (Insight 5).
- **`charge_tenure_ratio`** (4th, 0.0741) is fourth. This engineered feature captures high bills relative to tenure -- high bills for relatively new customers increase churn risk.
- **`has_long_contract`** (5th, 0.0364) is fifth. Having a long-term contract reduces churn risk.
- **`Contract`** (6th, 0.0159) is sixth. Month-to-month customers churn far more than those on longer contracts.

**Features ranked lower than expected:** `num_addon_services` ranks well below the top features, and `is_new_customer` ranks below `tenure`. This is expected because `is_new_customer` is derived from `tenure` (tenure <= 12), so `tenure` already captures the same information. The overlap explains why `is_new_customer` adds little incremental importance beyond `tenure`.

Each feature's observed direction matches the business expectation from Section 4.2: `Contract`, `tenure`, and `has_long_contract` reduce churn risk; `charge_tenure_ratio` and `MonthlyCharges` increase it."""
cells[56]["source"] = new56.splitlines(keepends=True)

# ---------- A2 + A6: Cell 60 - fix sensitivity formula and Rec 2 finding ----------
c60 = cells[60]
src = "".join(c60["source"])
src = src.replace(
    "New customers (tenure <= 12) mean tenure 18.0 vs 37.6",
    "New customers (tenure <= 12) churn at 47.4% vs 17.1% for longer-tenure customers"
)
src = src.replace(
    "net_impact * 0.15 / 0.30",
    "tp_contacted * 0.15 * avg_charges_churners * 12 - program_cost"
)
src = src.replace(
    "net_impact * 0.45 / 0.30",
    "tp_contacted * 0.45 * avg_charges_churners * 12 - program_cost"
)
c60["source"] = src.splitlines(keepends=True)

# ---------- Insert A7 (LR coefficients cell) after cell 56 ----------
a7_code = """# Direction check: Logistic Regression coefficients (A7)
clf = final_pipeline.named_steps["clf"]
prep = final_pipeline.named_steps["prep"]
feature_names = list(prep.get_feature_names_out())
coefs = clf.coef_[0]
coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})

print("=== Logistic Regression Coefficients (Direction Check) ===")
print()

for feat in ["tenure", "MonthlyCharges", "charge_tenure_ratio", "has_long_contract", "is_new_customer", "num_addon_services"]:
    row = coef_df[coef_df["feature"] == feat]
    if len(row) > 0:
        print(f"{feat}: coef = {row['coef'].values[0]:+.4f}")
    else:
        print(f"{feat}: not found (may have been removed in feature selection)")

print()
print("Contract (one-hot encoded coefficients):")
for feat in ["Contract_Month-to-month", "Contract_One year", "Contract_Two year"]:
    row = coef_df[coef_df["feature"] == feat]
    if len(row) > 0:
        print(f"  {feat}: coef = {row['coef'].values[0]:+.4f}")
    else:
        print(f"  {feat}: not found")

print()
print("Negative coef = lower churn risk; positive coef = higher churn risk.")
print("Expected: tenure(-), has_long_contract(-), is_new_customer(+), charge_tenure_ratio(+), MonthlyCharges(+), Contract_Month-to-month(+).")"""

a7_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": a7_code.splitlines(keepends=True)
}

# ---------- Insert A3 (error analysis interpretation) after cell 53 ----------
a3_md = """## Error Analysis Interpretation

**False negatives (188 missed churners):** 76.6% are month-to-month, with mean tenure of 25.2 months and mean MonthlyCharges of 67.65. These are customers the model failed to flag in the top-20%. Their relatively high tenure and charges suggest they are longer-tenured customers whose churn risk increased over time -- they fall outside the top-20% cutoff despite being valuable customers. To reduce FN, consider lowering the threshold for high-value customers (those with above-median tenure and charges) or adding a second-tier outreach list beyond the top-20%.

**False positives (96 contacted but did not churn):** 99.0% are month-to-month, with mean tenure of 11.9 months (very new customers) and mean MonthlyCharges of 80.35 (above average). These are new, high-spending month-to-month customers the model flagged as risky but who did not churn. Contacting them wastes part of the 282-customer budget. To reduce FP, consider requiring additional risk signals (e.g., recent service complaints or usage decline) before contacting new customers.

**One concrete change:** Add a "tenure x contract" interaction feature. Month-to-month customers with short tenure are the highest-risk group; flagging them specifically would reduce both FN (catch more of the 188 missed churners who are 76.6% month-to-month) and FP (avoid contacting long-tenured customers unlikely to churn)."""

a3_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": a3_md.splitlines(keepends=True)
}

# Insert from highest index to lowest so indices don't shift
cells.insert(57, a7_cell)   # after cell 56
cells.insert(54, a3_cell)   # after cell 53

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
print(f"Total cells: {len(nb['cells'])}")
