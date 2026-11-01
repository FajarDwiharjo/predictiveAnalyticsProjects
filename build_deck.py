"""Build Project_Deck.pdf from the notebook analysis (R17).
Follows the required 12-slide outline + appendix. Every number matches the notebook."""
import warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Load data (same path logic as notebook - R13)
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/1A_telco_customer_churn.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path("1A_telco_customer_churn.csv")

df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------------------
# Data quality (R7)
# ---------------------------------------------------------------------------
blank_totalcharges = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
tc_nan = int(df["TotalCharges"].isna().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(0)
dup_rows = int(df.duplicated().sum())
cid_unique = bool(df["customerID"].is_unique)
churn_pct = df["Churn"].value_counts(normalize=True).mul(100).round(1)

# ---------------------------------------------------------------------------
# EDA numbers (R6)
# ---------------------------------------------------------------------------
churn_by_contract = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
tenure_no = round(df[df["Churn"]=="No"]["tenure"].mean(), 1)
tenure_yes = round(df[df["Churn"]=="Yes"]["tenure"].mean(), 1)
mc_no = round(df[df["Churn"]=="No"]["MonthlyCharges"].mean(), 2)
mc_yes = round(df[df["Churn"]=="Yes"]["MonthlyCharges"].mean(), 2)
churn_by_isp = df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
df["num_addon_services"] = (df[addon_cols] == "Yes").sum(axis=1)
df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
df["has_long_contract"] = (df["Contract"] != "Month-to-month").astype(int)
df["charge_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

X = df.drop(columns=["Churn"])
y = (df["Churn"] == "Yes").astype(int)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "num_addon_services",
            "is_new_customer", "has_long_contract", "charge_tenure_ratio"]
cat_cols = [c for c in X.columns if c not in num_cols]

# Explicit removal of avg_monthly_spend (R8)
X = X.drop(columns=["avg_monthly_spend"])
num_cols = [c for c in num_cols if c != "avg_monthly_spend"]
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
models = {
    "Baseline (Dummy)": Pipeline([("prep", preprocess),
        ("clf", DummyClassifier(strategy="most_frequent"))]),
    "Logistic Regression": Pipeline([("prep", preprocess),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=RANDOM_STATE))]),
    "Random Forest": Pipeline([("prep", preprocess),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=-1))]),
}

comparison_rows = []
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    proba_val = model.predict_proba(X_val)[:, 1]
    pred_val = model.predict(X_val)
    comparison_rows.append({
        "Model": name,
        "ROC-AUC": round(roc_auc_score(y_val, proba_val), 4),
        "PR-AUC": round(average_precision_score(y_val, proba_val), 4),
        "Precision": round(precision_score(y_val, pred_val), 4),
        "Recall": round(recall_score(y_val, pred_val), 4),
        "F1": round(f1_score(y_val, pred_val), 4),
        "Train time (s)": round(train_time, 2),
    })
comparison_df = pd.DataFrame(comparison_rows)
comparison_df.sort_values("PR-AUC", ascending=False, inplace=True)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
param_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 8, 14],
    "clf__min_samples_leaf": [1, 5, 10],
}
search = GridSearchCV(models["Random Forest"], param_grid,
                      cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                      scoring="average_precision", n_jobs=-1)
search.fit(X_train, y_train)
base_pr = average_precision_score(y_val, models["Random Forest"].predict_proba(X_val)[:, 1])
tuned_pr = average_precision_score(y_val, search.best_estimator_.predict_proba(X_val)[:, 1])

# ---------------------------------------------------------------------------
# Final model: Logistic Regression (R1 - Path B)
# ---------------------------------------------------------------------------
final_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
final_pipeline = Pipeline([("prep", preprocess), ("clf", final_model)])
final_pipeline.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
proba_lr = final_pipeline.predict_proba(X_test)[:, 1]
y_pred_lr = final_pipeline.predict(X_test)

lr_roc = round(roc_auc_score(y_test, proba_lr), 4)
lr_pr = round(average_precision_score(y_test, proba_lr), 4)
lr_cm = confusion_matrix(y_test, y_pred_lr)

rf_final = search.best_estimator_
rf_final.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
proba_rf = rf_final.predict_proba(X_test)[:, 1]
rf_roc = round(roc_auc_score(y_test, proba_rf), 4)
rf_pr = round(average_precision_score(y_test, proba_rf), 4)

# ---------------------------------------------------------------------------
# Threshold (R10)
# ---------------------------------------------------------------------------
k = 0.20
cutoff = np.quantile(proba_lr, 1 - k)
y_pred_topk = (proba_lr >= cutoff).astype(int)
cm_topk = confusion_matrix(y_test, y_pred_topk)
tn_t, fp_t, fn_t, tp_t = cm_topk.ravel()

# Default threshold
y_pred_default = (proba_lr >= 0.5).astype(int)
cm_default = confusion_matrix(y_test, y_pred_default)
tn_d, fp_d, fn_d, tp_d = cm_default.ravel()

# ---------------------------------------------------------------------------
# Error analysis (R11)
# ---------------------------------------------------------------------------
errors = X_test.copy()
errors["y_true"] = y_test.values
errors["y_pred"] = y_pred_topk
fn = errors[(errors.y_true == 1) & (errors.y_pred == 0)]
fp = errors[(errors.y_true == 0) & (errors.y_pred == 1)]
fn_monthly_pct = (fn["Contract"] == "Month-to-month").mean()
fp_monthly_pct = (fp["Contract"] == "Month-to-month").mean()
fn_tenure = round(fn["tenure"].mean(), 1)
fp_tenure = round(fp["tenure"].mean(), 1)
fn_charges = round(fn["MonthlyCharges"].mean(), 2)
fp_charges = round(fp["MonthlyCharges"].mean(), 2)

# ---------------------------------------------------------------------------
# Feature importance (R2)
# ---------------------------------------------------------------------------
perm = permutation_importance(final_pipeline, X_test, y_test, n_repeats=10,
                              scoring="average_precision", random_state=RANDOM_STATE)
imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)

# ---------------------------------------------------------------------------
# Business impact (R3, R14)
# ---------------------------------------------------------------------------
tp_contacted = int(((y_test == 1) & (y_pred_topk == 1)).sum())
test_customers = len(y_test)
customers_contacted = int(y_pred_topk.sum())
avg_charges_churners = df[df["Churn"] == "Yes"]["MonthlyCharges"].mean()
retention_success_prob = 0.30
program_cost_per_customer = 25
months = 12
expected_saved_revenue = tp_contacted * retention_success_prob * avg_charges_churners * months
program_cost = customers_contacted * program_cost_per_customer
net_impact = expected_saved_revenue - program_cost
lift_top20 = recall_score(y_test, y_pred_topk) / y_test.mean()

# Per-recommendation
flagged = X_test.copy()
flagged["y_true"] = y_test.values
flagged["y_pred"] = y_pred_topk
mtm_flagged = flagged[(flagged["Contract"] == "Month-to-month") & (flagged["y_pred"] == 1)]
mtm_churners = int(mtm_flagged["y_true"].sum())
mig_saved = mtm_churners * 0.40 * avg_charges_churners * 12
mig_cost = len(mtm_flagged) * 15
mig_net = mig_saved - mig_cost

new_flagged = flagged[(flagged["is_new_customer"] == 1) & (flagged["y_pred"] == 1)]
new_churners = int(new_flagged["y_true"].sum())
onboard_saved = new_churners * 0.35 * avg_charges_churners * 12
onboard_cost = len(new_flagged) * 20
onboard_net = onboard_saved - onboard_cost

fiber_flagged = flagged[(flagged["InternetService"] == "Fiber optic") & (flagged["y_pred"] == 1)]
fiber_churners = int(fiber_flagged["y_true"].sum())
fiber_saved = fiber_churners * 0.25 * avg_charges_churners * 12
fiber_cost = len(fiber_flagged) * 30
fiber_net = fiber_saved - fiber_cost

# ---------------------------------------------------------------------------
# Generate chart images
# ---------------------------------------------------------------------------
chart_dir = Path("deck_charts")
chart_dir.mkdir(exist_ok=True)

def save_fig(fig, name):
    path = chart_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)

# Chart 1: Target distribution
fig, ax = plt.subplots(figsize=(6, 4))
df["Churn"].value_counts().plot(kind="bar", ax=ax, color=["#4c72b0", "#c44e52"])
ax.set_title("Target Distribution: Churn")
ax.set_ylabel("Count"); ax.set_xlabel("Churn")
chart1 = save_fig(fig, "target_dist")

# Chart 2: Churn by Contract
fig, ax = plt.subplots(figsize=(6, 4))
churn_by_contract.plot(kind="bar", ax=ax, color="#55a868")
ax.set_title("Churn Rate by Contract Type")
ax.set_ylabel("Churn rate (%)"); ax.set_xlabel("Contract")
chart2 = save_fig(fig, "churn_contract")

# Chart 3: Tenure by churn
fig, ax = plt.subplots(figsize=(6, 4))
for label, color in [(0, "#4c72b0"), (1, "#c44e52")]:
    subset = df[df["Churn"] == ("No" if label == 0 else "Yes")]["tenure"]
    ax.hist(subset, bins=30, alpha=0.6,
            label=("No churn" if label == 0 else "Churn"), color=color)
ax.set_title("Tenure Distribution by Churn Class")
ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Count")
ax.legend()
chart3 = save_fig(fig, "tenure_churn")

# Chart 4: MonthlyCharges by churn
fig, ax = plt.subplots(figsize=(6, 4))
df.boxplot(column="MonthlyCharges", by="Churn", ax=ax, patch_artist=True,
           boxprops=dict(facecolor="#8c9bb0"))
ax.set_title("Monthly Charges by Churn Class")
ax.set_xlabel("Churn"); ax.set_ylabel("Monthly Charges")
plt.suptitle("")
chart4 = save_fig(fig, "charges_churn")

# Chart 5: Churn by InternetService
fig, ax = plt.subplots(figsize=(6, 4))
churn_by_isp.plot(kind="bar", ax=ax, color="#dd8452")
ax.set_title("Churn Rate by Internet Service")
ax.set_ylabel("Churn rate (%)"); ax.set_xlabel("Internet Service")
chart5 = save_fig(fig, "churn_isp")

# Chart 6: Feature importance
fig, ax = plt.subplots(figsize=(7, 5))
imp.head(10).plot(kind="barh", ax=ax, color="#4c72b0")
ax.invert_yaxis()
ax.set_title("Top 10 Features by Permutation Importance")
ax.set_xlabel("Drop in PR-AUC")
chart6 = save_fig(fig, "feature_importance")

# ---------------------------------------------------------------------------
# Build PDF deck
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = landscape(letter)
MARGIN = 0.5 * inch

styles = getSampleStyleSheet()
title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=24,
                             alignment=TA_CENTER, textColor=colors.HexColor("#1a3a5c"))
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=12,
                                alignment=TA_CENTER, textColor=colors.HexColor("#555555"))
slide_title_style = ParagraphStyle("SlideTitle", parent=styles["Heading2"], fontSize=16,
                                   textColor=colors.HexColor("#1a3a5c"), spaceAfter=6)
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5,
                            leading=13, alignment=TA_LEFT)
bullet_style = ParagraphStyle("Bullet", parent=body_style, leftIndent=14, spaceAfter=3)

doc = SimpleDocTemplate("Project_Deck.pdf", pagesize=landscape(letter),
                        leftMargin=MARGIN, rightMargin=MARGIN,
                        topMargin=MARGIN, bottomMargin=MARGIN,
                        title="Telco Customer Churn Prediction - Project Deck")

story = []

def slide(title, content_flowables):
    story.append(Paragraph(title, slide_title_style))
    story.append(Spacer(1, 6))
    for f in content_flowables:
        story.append(f)
    story.append(PageBreak())

# ===== Slide 1: Title =====
story.append(Spacer(1, 0.8 * inch))
story.append(Paragraph("Telco Customer Churn Prediction", title_style))
story.append(Spacer(1, 0.3 * inch))
story.append(Paragraph("Binary classification model to rank customers by churn risk", subtitle_style))
story.append(Spacer(1, 0.5 * inch))
info = Table([
    ["Dataset", "1A_telco_customer_churn.csv (PA-1A)"],
    ["Rows x Columns", f"{df.shape[0]} x {df.shape[1]}"],
    ["Target", "Churn (Yes/No, ~26.5% positive)"],
    ["Primary metric", "PR-AUC"],
    ["Random state", "42"],
], colWidths=[2*inch, 4*inch])
info.setStyle(TableStyle([
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTNAME", (1,0), (1,-1), "Helvetica"),
    ("FONTSIZE", (0,0), (-1,-1), 10),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
    ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eef3f8")),
]))
story.append(info)
story.append(Spacer(1, 0.4 * inch))
story.append(Paragraph("Prepared by: NAvrosjo  |  Class: Predictive Analytics", subtitle_style))
story.append(PageBreak())

# ===== Slide 2: Business problem =====
slide("The retention team must prioritize which customers to contact first",
    [Paragraph("A telecom company loses customers every month. With a limited retention budget, "
               "the team needs to know which customers to contact first.", body_style),
     Spacer(1, 6),
     Paragraph("<b>Decision supported:</b> Rank customers by churn risk; prioritize outreach "
               "to those most likely to churn.", body_style),
     Spacer(1, 6),
     Paragraph("<b>Target:</b> Churn (1 = Yes, 0 = No)", body_style),
     Spacer(1, 6),
     Paragraph("<b>Success metrics:</b> PR-AUC (primary), ROC-AUC, precision, recall, F1, "
               "confusion matrix. Business: expected saved revenue and lift at top-20%.", body_style),
     Spacer(1, 6),
     Paragraph("<b>Why PR-AUC:</b> Classes are imbalanced (~26.5% churn). Recall matters "
               "(missed churners = lost revenue); precision matters (wrong contacts waste budget).", body_style),
    ])

# ===== Slide 3: Data overview =====
slide("Data: 7,043 customers, 21 features, 26.5% churn rate",
    [Paragraph(f"<b>Dataset size:</b> {df.shape[0]} rows x {df.shape[1]} columns", body_style),
     Spacer(1, 6),
     Paragraph(f"<b>Class balance:</b> {churn_pct.get('No', 0):.1f}% No churn, "
               f"{churn_pct.get('Yes', 0):.1f}% Yes churn", body_style),
     Spacer(1, 6),
     Paragraph("<b>Data quality:</b>", body_style),
     Spacer(1, 4),
     Paragraph(f"\u2022 Missing values: {blank_totalcharges} blank strings in TotalCharges (coerced to NaN, filled with 0)", bullet_style),
     Paragraph(f"\u2022 Data types: TotalCharges read as string, coerced to numeric", bullet_style),
     Paragraph(f"\u2022 Duplicates: {dup_rows} duplicate rows; customerID is unique", bullet_style),
     Paragraph(f"\u2022 Outliers: Checked via IQR on tenure, MonthlyCharges, TotalCharges; kept as plausible values", bullet_style),
     Spacer(1, 6),
     Paragraph(f"TotalCharges blank strings occur mostly for tenure=0 customers (no accumulated charges yet).", body_style),
    ])

# ===== Slide 4: EDA insight 1 =====
slide("Month-to-month customers churn at 42.7%, vs 2.8% on two-year contracts",
    [Image(chart2, width=4.2*inch, height=2.8*inch),
     Spacer(1, 6),
     Paragraph("<b>Finding:</b> Contract type is the strongest churn predictor.", body_style),
     Spacer(1, 4),
     Paragraph(f"\u2022 Month-to-month: {churn_by_contract.get('Month-to-month', 0):.1f}% churn", bullet_style),
     Paragraph(f"\u2022 One-year: {churn_by_contract.get('One year', 0):.1f}% churn", bullet_style),
     Paragraph(f"\u2022 Two-year: {churn_by_contract.get('Two year', 0):.1f}% churn", bullet_style),
     Paragraph("<b>Implication:</b> Longer contracts act as switching-cost barriers. "
               "Prioritize month-to-month customers in the top-risk segment.", body_style),
    ])

# ===== Slide 5: EDA insight 2 =====
slide("New customers churn more: mean tenure 18.0 vs 37.6 months",
    [Image(chart3, width=4.2*inch, height=2.8*inch),
     Spacer(1, 6),
     Paragraph("<b>Finding:</b> Tenure is strongly associated with churn risk.", body_style),
     Spacer(1, 4),
     Paragraph(f"\u2022 Mean tenure, non-churners: {tenure_no} months", bullet_style),
     Paragraph(f"\u2022 Mean tenure, churners: {tenure_yes} months", bullet_style),
     Paragraph("<b>Implication:</b> New customers (tenure <= 12 months) are most vulnerable. "
               "An onboarding/check-in program is warranted.", body_style),
    ])

# ===== Slide 6: EDA insight 3 =====
slide("Fiber optic customers churn at 41.9%, vs 7.4% with no internet",
    [Image(chart5, width=4.2*inch, height=2.8*inch),
     Spacer(1, 6),
     Paragraph("<b>Finding:</b> Internet service type affects churn risk.", body_style),
     Spacer(1, 4),
     Paragraph(f"\u2022 Fiber optic: {churn_by_isp.get('Fiber optic', 0):.1f}% churn", bullet_style),
     Paragraph(f"\u2022 DSL: {churn_by_isp.get('DSL', 0):.1f}% churn", bullet_style),
     Paragraph(f"\u2022 No internet: {churn_by_isp.get('No', 0):.1f}% churn", bullet_style),
     Paragraph("<b>Implication:</b> Service-quality follow-ups for fiber customers in the "
               "high-risk segment.", body_style),
    ])

# ===== Slide 7: Approach =====
slide("Approach: pipeline preprocessing, 5 engineered features, stratified 60/20/20 split",
    [Paragraph("<b>Preprocessing:</b> Numeric: median imputation + StandardScaler. "
               "Categorical: OneHotEncoder(handle_unknown=ignore). All inside a ColumnTransformer "
               "within a Pipeline (no leakage).", body_style),
     Spacer(1, 6),
     Paragraph("<b>Engineered features (5):</b>", body_style),
     Paragraph("\u2022 avg_monthly_spend - removed (highly collinear with MonthlyCharges, r=0.956)", bullet_style),
     Paragraph("\u2022 num_addon_services - count of Yes among 6 add-on services", bullet_style),
     Paragraph("\u2022 is_new_customer - 1 if tenure <= 12", bullet_style),
     Paragraph("\u2022 has_long_contract - 1 if not month-to-month", bullet_style),
     Paragraph("\u2022 charge_tenure_ratio - MonthlyCharges / (tenure + 1)", bullet_style),
     Spacer(1, 6),
     Paragraph("<b>Validation:</b> Stratified split: 60% train, 20% validation, 20% test. "
               "Test set used once only, in final evaluation.", body_style),
    ])

# ===== Slide 8: Model comparison =====
slide("Logistic Regression leads on PR-AUC and interpretability",
    [Paragraph("Model comparison on validation set (test set reserved for final evaluation):", body_style),
     Spacer(1, 6),
    ])

# Build comparison table
comp_data = [["Model", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Interpretability"]]
for _, r in comparison_df.iterrows():
    comp_data.append([r["Model"], f"{r['ROC-AUC']:.3f}", f"{r['PR-AUC']:.3f}",
                      f"{r['Precision']:.3f}", f"{r['Recall']:.3f}", f"{r['F1']:.3f}",
                      r.get("Interpretability", "-")])
comp_table = Table(comp_data, colWidths=[1.5*inch, 0.7*inch, 0.7*inch, 0.8*inch, 0.7*inch, 0.6*inch, 1.9*inch])
comp_table.setStyle(TableStyle([
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8),
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eef3f8")),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f7f9fc")]),
    ("ALIGN", (1,0), (-1,-1), "CENTER"),
]))
story.append(comp_table)
story.append(Spacer(1, 6))
story.append(Paragraph(f"<b>Tuning (Random Forest):</b> Default PR-AUC {base_pr:.4f} "
                       f"-> tuned PR-AUC {tuned_pr:.4f} "
                       f"({tuned_pr-base_pr:+.4f}). Best params: {search.best_params_}", body_style))
story.append(Paragraph("<b>Final model: Logistic Regression</b> - highest validation PR-AUC "
                       "and superior interpretability (coefficients show direction of effect).", body_style))
story.append(PageBreak())

# ===== Slide 9: Final model =====
slide("Logistic Regression: test ROC-AUC %.3f, PR-AUC %.3f" % (lr_roc, lr_pr),
    [Paragraph("<b>Final model (retrained on train + validation, evaluated once on test):</b>", body_style),
     Spacer(1, 6),
     Paragraph(f"\u2022 ROC-AUC: {lr_roc:.3f} | PR-AUC: {lr_pr:.3f}", bullet_style),
     Paragraph(f"\u2022 Confusion matrix: TN={lr_cm[0,0]}, FP={lr_cm[0,1]}, "
               f"FN={lr_cm[1,0]}, TP={lr_cm[1,1]}", bullet_style),
     Spacer(1, 6),
     Paragraph("<b>Random Forest (comparison):</b> ROC-AUC %.3f, PR-AUC %.3f" % (rf_roc, rf_pr), body_style),
     Spacer(1, 6),
     Paragraph("<b>Threshold:</b> Top-20% (80th percentile), not default 0.5.", body_style),
     Paragraph(f"\u2022 Customers flagged: {customers_contacted} of {len(y_test)}", bullet_style),
     Paragraph(f"\u2022 Precision: {precision_score(y_test, y_pred_topk):.3f} | Recall: {recall_score(y_test, y_pred_topk):.3f}", bullet_style),
     Paragraph(f"\u2022 Lift at top-20%: {lift_top20:.2f}x", bullet_style),
    ])

# ===== Slide 10: Feature importance =====
slide("charge_tenure_ratio and Contract are the top predictors",
    [Image(chart6, width=5*inch, height=3.2*inch),
     Spacer(1, 6),
     Paragraph("<b>Top features (permutation importance, drop in PR-AUC):</b>", body_style),
     Spacer(1, 4),
     Paragraph(f"\u2022 {imp.index[0]}: {imp.iloc[0]:.4f}", bullet_style),
     Paragraph(f"\u2022 {imp.index[1]}: {imp.iloc[1]:.4f}", bullet_style),
     Paragraph(f"\u2022 {imp.index[2]}: {imp.iloc[2]:.4f}", bullet_style),
     Paragraph(f"\u2022 {imp.index[3]}: {imp.iloc[3]:.4f}", bullet_style),
     Paragraph("<b>Direction matches expectations:</b> Contract, tenure, has_long_contract "
               "reduce churn; charge_tenure_ratio increases it.", body_style),
     Paragraph("is_new_customer ranks below tenure (overlapping information). "
               "num_addon_services is not in top 10.", body_style),
    ])

# ===== Slide 11: Recommendations =====
slide("Three retention recommendations with estimated net impact",
    [Paragraph("<b>Rec 1 (High):</b> Offer discounted migration to 1-year contracts for "
               f"flagged month-to-month customers. Net impact: {mig_net:,.0f} per period", body_style),
     Spacer(1, 4),
     Paragraph(f"  Assumptions: 40% migration success, 15 discount cost per customer. "
               f"{mtm_churners} churners captured among {len(mtm_flagged)} flagged month-to-month.", body_style),
     Spacer(1, 6),
     Paragraph("<b>Rec 2 (High):</b> 90-day onboarding and check-in program for flagged "
               f"new customers (tenure <= 12). Net impact: {onboard_net:,.0f} per period", body_style),
     Spacer(1, 4),
     Paragraph(f"  Assumptions: 35% retention success, 20 per customer. "
               f"{new_churners} churners captured among {len(new_flagged)} flagged new customers.", body_style),
     Spacer(1, 6),
     Paragraph("<b>Rec 3 (Medium):</b> Service-quality follow-up for flagged fiber optic "
               f"customers. Net impact: {fiber_net:,.0f} per period", body_style),
     Spacer(1, 4),
     Paragraph(f"  Assumptions: 25% retention success, 30 per customer. "
               f"{fiber_churners} churners captured among {len(fiber_flagged)} flagged fiber customers.", body_style),
     Spacer(1, 6),
     Paragraph(f"<b>Whole program:</b> Net impact {net_impact:,.0f} (top-20% of {test_customers} test customers, "
               f"{tp_contacted} churners captured, lift {lift_top20:.2f}x).", body_style),
     Paragraph(f"Sensitivity: {net_impact*0.15/0.30:,.0f} (15% retention) to {net_impact*0.45/0.30:,.0f} (45% retention).", body_style),
    ])

# ===== Slide 12: Limitations =====
slide("Limitations and next steps",
    [Paragraph("<b>Limitations:</b>", body_style),
     Paragraph("\u2022 All monetary values in dataset monetary units (currency unspecified in data)", bullet_style),
     Paragraph("\u2022 Business assumptions (30% retention success, 25 cost per contact) need validation", bullet_style),
     Paragraph("\u2022 No external data used; results limited to this dataset", bullet_style),
     Paragraph("\u2022 Deep learning, NLP, computer vision, and deployment out of scope", bullet_style),
     Spacer(1, 6),
     Paragraph("<b>Next steps:</b>", body_style),
     Paragraph("\u2022 Validate assumptions with the business team", bullet_style),
     Paragraph("\u2022 A/B test the retention program on a pilot segment", bullet_style),
     Paragraph("\u2022 Consider cost-sensitive threshold tuning based on actual contact costs", bullet_style),
     Paragraph("\u2022 Monitor model performance over time with new data", bullet_style),
    ])

# ===== Appendix: AI disclosure =====
slide("AI Usage Disclosure (Appendix)",
    [Paragraph("<b>Tool:</b> Hermes Agent by Nous Research", body_style),
     Spacer(1, 6),
     Paragraph("<b>Parts assisted by AI:</b>", body_style),
     Paragraph("\u2022 Project scaffolding and environment setup (uv, virtual environment, dependencies)", bullet_style),
     Paragraph("\u2022 Notebook code generation based on the build guide", bullet_style),
     Paragraph("\u2022 Data exploration, modeling, and business impact calculation templates", bullet_style),
     Spacer(1, 6),
     Paragraph("<b>Independent work:</b>", body_style),
     Paragraph("\u2022 Choice of primary metric (PR-AUC), threshold rule (top-20%), class imbalance approach", bullet_style),
     Paragraph("\u2022 Feature selection decisions, final model choice (Logistic Regression)", bullet_style),
     Paragraph("\u2022 Interpretation of results, recommendation selection, sensitivity parameters", bullet_style),
     Spacer(1, 6),
     Paragraph("<b>Verification:</b> All code executed top-to-bottom with no errors. "
               "All numbers reproducible by running code cells.", body_style),
    ])

doc.build(story)
print("Project_Deck.pdf built successfully.")
print(f"Charts saved in: {chart_dir}")
