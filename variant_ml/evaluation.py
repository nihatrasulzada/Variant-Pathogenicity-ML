"""
ADDIM 4a: Nəticələrin qiymətləndirilməsi.

  - Accuracy, Precision, Recall, F1-Score, ROC-AUC (cədvəl + CSV)
  - ROC əyriləri (hər iki model eyni qrafikdə)
  - Qarışıqlıq matrisləri (confusion matrix)
  - Feature importance (ən vacib N xüsusiyyət)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ekran olmayan mühitdə də işləsin
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from . import config
from .models import get_feature_names

sns.set_theme(style="whitegrid", context="notebook")
MODEL_COLORS = {"Random Forest": "#2a9d8f", "XGBoost": "#e76f51"}

# İnsan oxuya bilən xüsusiyyət adları (qrafiklər üçün)
_PRETTY = {
    "cadd_phred": "CADD (PHRED)", "polyphen_score": "PolyPhen-2", "sift_score": "SIFT",
    "blosum62": "BLOSUM62", "in_functional_domain": "Funksional domendə",
    "abs_delta_hydrophobicity": "|Δ hidrofobluq|", "abs_delta_volume": "|Δ həcm|",
    "delta_charge": "Δ yük", "charge_change": "Yük dəyişməsi",
}


def compute_metrics(models: dict, X_test, y_test, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba),
        })
    return pd.DataFrame(rows).set_index("Model").round(4)


def plot_roc_curves(models: dict, X_test, y_test, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, lw=2.2, color=MODEL_COLORS.get(name),
                label=f"{name} (AUC = {roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Təsadüfi (AUC = 0.5)")
    ax.set(xlabel="False Positive Rate (1 − Spesifiklik)",
           ylabel="True Positive Rate (Həssaslıq)",
           title="ROC əyriləri — sınaq dataseti", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_confusion_matrices(models: dict, X_test, y_test, out: Path) -> Path:
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4))
    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Benign", "Pathogenic"],
                    yticklabels=["Benign", "Pathogenic"])
        ax.set(title=name, xlabel="Proqnoz", ylabel="Həqiqi")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def feature_importance_table(model, input_columns: list[str]) -> pd.Series:
    names = get_feature_names(model, input_columns)
    imp = model.named_steps["clf"].feature_importances_
    return pd.Series(imp, index=names).sort_values(ascending=False)


def plot_feature_importance(models: dict, input_columns: list[str], out: Path,
                            top_n: int = 15) -> Path:
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6.5))
    for ax, (name, model) in zip(axes, models.items()):
        imp = feature_importance_table(model, input_columns).head(top_n)[::-1]
        labels = [_PRETTY.get(i, i) for i in imp.index]
        ax.barh(labels, imp.values, color=MODEL_COLORS.get(name))
        ax.set(title=f"{name} — top {top_n} xüsusiyyət", xlabel="Vaciblik (importance)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def evaluate_models(models: dict, X_test, y_test, input_columns: list[str],
                    out_dir: Path = config.OUTPUT_DIR) -> pd.DataFrame:
    """Bütün metrikaları hesablayır, çap edir və qrafikləri `outputs/` qovluğuna yazır."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(models, X_test, y_test)
    metrics.to_csv(out_dir / "metrics.csv")

    print("\n================ Sınaq nəticələri ================")
    print(metrics.to_string())
    print("==================================================\n")

    paths = [
        plot_roc_curves(models, X_test, y_test, out_dir / "roc_curves.png"),
        plot_confusion_matrices(models, X_test, y_test, out_dir / "confusion_matrices.png"),
        plot_feature_importance(models, input_columns, out_dir / "feature_importance.png"),
    ]
    for name, model in models.items():
        feature_importance_table(model, input_columns).to_csv(
            out_dir / f"feature_importance_{name.lower().replace(' ', '_')}.csv",
            header=["importance"])
    for p in paths:
        print(f"[eval] Qrafik saxlanıldı: {p}")
    return metrics
