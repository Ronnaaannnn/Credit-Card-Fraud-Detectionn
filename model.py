"""Training pipeline: 70/30 split, Random Forest, metrics + graphs."""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts")
PLOTS = os.path.join(BASE, "static", "plots")
MODEL_PATH = os.path.join(ART, "model.joblib")
METRICS_PATH = os.path.join(ART, "metrics.json")
SAMPLES_PATH = os.path.join(ART, "test_samples.csv")

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TIME_MAX = 172792  # last second in the dataset (~48 hours)
THRESHOLD = 0.5

BG, FG, ACC, ACC2 = "#0f172a", "#e2e8f0", "#38bdf8", "#f43f5e"


def _style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=13, pad=10)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color("#334155")
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)


def _save(fig, name):
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, name), dpi=110, facecolor=BG)
    plt.close(fig)


def train(csv_path):
    os.makedirs(ART, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[(df["Time"] >= 0) & (df["Time"] <= TIME_MAX)]  # time range 0..172792
    X, y = df[FEATURES], df["Class"]

    # 70 / 30 split, stratified so frauds appear in both parts
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=14, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_te, pred, labels=[0, 1])
    auprc = average_precision_score(y_te, proba)
    auc = roc_auc_score(y_te, proba)

    metrics = {
        "rows": int(len(df)), "frauds": int(y.sum()),
        "fraud_pct": round(100 * float(y.mean()), 3),
        "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
        "train_frauds": int(y_tr.sum()), "test_frauds": int(y_te.sum()),
        "auprc": round(float(auprc), 4), "roc_auc": round(float(auc), 4),
        "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "cm": cm.tolist(), "threshold": THRESHOLD,
        "report": classification_report(y_te, pred, digits=4, zero_division=0),
    }

    # ---------- graphs ----------
    # 1. class distribution (log scale, since it is so unbalanced)
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = y.value_counts().sort_index()
    ax.bar(["Legit", "Fraud"], counts.values, color=[ACC, ACC2])
    ax.set_yscale("log")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", color=FG)
    _style(ax, "Class distribution (log scale)")
    _save(fig, "class_dist.png")

    # 2. precision-recall curve
    p, r, _ = precision_recall_curve(y_te, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p, color=ACC, lw=2, label=f"AUPRC = {auprc:.3f}")
    ax.axhline(y_te.mean(), color="#64748b", ls="--", label="No-skill")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor="#334155")
    _style(ax, "Precision-Recall curve")
    _save(fig, "pr_curve.png")

    # 3. ROC curve
    fpr, tpr, _ = roc_curve(y_te, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color=ACC2, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#64748b", ls="--")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor="#334155")
    _style(ax, "ROC curve")
    _save(fig, "roc_curve.png")

    # 4. confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Legit", "Fraud"]); ax.set_yticks([0, 1], ["Legit", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=14)
    _style(ax, "Confusion matrix (30% test set)")
    _save(fig, "confusion.png")

    # 5. feature importance
    imp = pd.Series(clf.feature_importances_, index=FEATURES).nlargest(15)[::-1]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.barh(imp.index, imp.values, color=ACC)
    _style(ax, "Top 15 feature importances")
    _save(fig, "importance.png")

    # 6. fraud over time + amount distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(df.loc[df.Class == 1, "Time"] / 3600, bins=48, color=ACC2)
    ax.set_xlabel("Hours since first transaction")
    ax.set_ylabel("Frauds")
    _style(ax, "Frauds over time (0 – 172792 s)")
    _save(fig, "fraud_time.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    bins = np.logspace(0, np.log10(max(df.Amount.max(), 10)), 40)
    ax.hist(df.loc[df.Class == 0, "Amount"].clip(lower=1), bins=bins, alpha=.7,
            color=ACC, label="Legit", density=True)
    ax.hist(df.loc[df.Class == 1, "Amount"].clip(lower=1), bins=bins, alpha=.7,
            color=ACC2, label="Fraud", density=True)
    ax.set_xscale("log"); ax.set_xlabel("Amount"); ax.set_ylabel("Density")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor="#334155")
    _style(ax, "Transaction amount")
    _save(fig, "amount.png")

    # ---------- persist ----------
    joblib.dump(clf, MODEL_PATH)
    test = X_te.assign(Class=y_te)
    keep = pd.concat([test[test.Class == 1], test[test.Class == 0].head(500)])
    keep.to_csv(SAMPLES_PATH, index=False)  # used by the "load example" buttons
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)
    return metrics


def load():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH)):
        return None, None
    with open(METRICS_PATH) as f:
        return joblib.load(MODEL_PATH), json.load(f)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "creditcard.csv")
    m = train(path)
    print(f"AUPRC={m['auprc']}  ROC-AUC={m['roc_auc']}  "
          f"P={m['precision']} R={m['recall']} F1={m['f1']}")
    print(m["report"])
