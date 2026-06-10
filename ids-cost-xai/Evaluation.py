"""
Evaluation utilities: TEL reporting, cost-ROC curves,
Spearman rank correlation between SHAP importance rankings.
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy.stats import spearmanr
from scipy.stats import rankdata

from logger import Logger
from Costs import total_economic_loss, CLASS_TO_IDX
 
log = Logger()

DEFAULT_REGIME = "moderate"
MACRO_CLASSES  = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']
CLASS_COLORS   = {'Normal': '#4CAF50', 'DoS': '#F44336',
                  'Probe': '#FF9800', 'R2L': '#9C27B0', 'U2R': '#F50057'}
FIGURE_DIR = "outputs/figures"

def show_classification_report(y_true, y_pred, le, title=""):
    if title:
        log.info(f"{title.upper()}")
    log.info(classification_report(y_true, y_pred, target_names=le.classes_))

def show_confusion_matrix(y_true_enc, y_pred_enc, le):
    cm = confusion_matrix(y_true_enc, y_pred_enc)
    df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    df.index.name = "True \\ Pred"
    log.info("Confusion Matrix: ")
    log.info(df.to_string())
    return cm

def report_tel(y_true_enc, y_pred_enc, model_name="Model", regime=DEFAULT_REGIME):
    """Print TEL breakdown and return the result dict."""
    result = total_economic_loss(y_true_enc, y_pred_enc)
    print(f"\n{'─'*50}")
    print(f"  Total Economic Loss — {model_name} [{regime}]")
    print(f"{'─'*50}")
    print(f"  TEL (total):     ${result['TEL']:>15,.2f}")
    print(f"  TEL from FN:     ${result['TEL_FN']:>15,.2f}")
    print(f"  TEL from FP:     ${result['TEL_FP']:>15,.2f}")
    print(f"\n  FN breakdown:")
    for cls, cost in result['FN_breakdown'].items():
        print(f"    {cls:8s}:     ${cost:>15,.2f}")
    return result
 
 
def compare_tel(tel_results: dict, regime=DEFAULT_REGIME, save=True):
    """
    Bar chart comparing TEL across models.
 
    tel_results: {model_name: tel_dict_from_report_tel()}
    """
    names  = list(tel_results.keys())
    totals = [tel_results[n]["TEL"] for n in names]
    fn_    = [tel_results[n]["TEL_FN"] for n in names]
    fp_    = [tel_results[n]["TEL_FP"] for n in names]
 
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_fn = ax.bar(x, fn_, label="FN cost", color="#F44336", alpha=0.85)
    bars_fp = ax.bar(x, fp_, bottom=fn_, label="FP cost", color="#FF9800", alpha=0.85)
 
    for i, total in enumerate(totals):
        ax.text(x[i], total + total * 0.01, f"${total:,.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
 
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Total Economic Loss (USD)")
    ax.set_title(f"TEL Comparison — {regime.capitalize()} Regime", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        import os; os.makedirs(FIGURE_DIR, exist_ok=True)
        plt.savefig(f"{FIGURE_DIR}/tel_comparison_{regime}.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_cost_roc(models: dict, X_test, y_test_enc, le,
                  focus_classes=("R2L", "U2R"), save=True):
    """
    Per-class ROC curves for high-severity minority classes.
    models: {label: fitted calibrated model}
    """
    import os; os.makedirs(FIGURE_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, len(focus_classes), figsize=(7 * len(focus_classes), 5))
    if len(focus_classes) == 1:
        axes = [axes]
 
    for ax, cls_name in zip(axes, focus_classes):
        cls_idx = CLASS_TO_IDX[cls_name]
        y_bin = (y_test_enc == cls_idx).astype(int)
 
        for model_name, model in models.items():
            probs     = model.predict_proba(X_test)[:, cls_idx]
            fpr, tpr, _ = roc_curve(y_bin, probs)
            roc_auc   = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.8, label=f"{model_name} (AUC={roc_auc:.3f})")
 
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title(f"Cost-ROC — {cls_name} Detection", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
 
    plt.tight_layout()
    if save:
        plt.savefig(f"{FIGURE_DIR}/cost_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
 
 
def spearman_shap_correlation(
    shap_baseline: np.ndarray,
    shap_costsens: np.ndarray,
    feature_names: list,
    le,
    save=True,
):
    """
    Compute Spearman rank correlation between baseline and cost-sensitive
    SHAP feature importance rankings, per class.
 
    A low ρ indicates a genuine 'attentional shift' toward security-critical
    features which is the core novelty metric of the proposal.
 
    shap_baseline / shap_costsens: (n_samples, n_features, n_classes)
    """
    import os; os.makedirs(FIGURE_DIR, exist_ok=True)
    n_classes = shap_baseline.shape[2]
    results = []
 
    print(f"\n{'─'*60}")
    print("  Spearman ρ — Baseline vs Cost-Sensitive SHAP Rankings")
    print(f"{'─'*60}")
    print(f"  {'Class':<10} {'ρ':>8}  {'p-value':>10}  {'Interpretation'}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*9}  {'─'*25}")
 
    for cls_idx in range(n_classes):
        cls_name = le.inverse_transform([cls_idx])[0]
        imp_base = np.abs(shap_baseline[:, :, cls_idx]).mean(axis=0)
        imp_cost = np.abs(shap_costsens[:, :, cls_idx]).mean(axis=0)
        rho, p   = spearmanr(rankdata(-imp_base), rankdata(-imp_cost))
        interp   = "strong shift ✓" if rho < 0.5 else ("moderate shift" if rho < 0.75 else "similar ranking")
        results.append({"class": cls_name, "rho": rho, "p_value": p})
        print(f"  {cls_name:<10} {rho:>8.3f}  {p:>10.4f}  {interp}")
 
    # Bar chart of ρ per class
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [CLASS_COLORS.get(c, "#607D8B") for c in df["class"]]
    ax.barh(df["class"], df["rho"], color=colors, edgecolor="white")
    ax.axvline(0.75, color="gray", linestyle="--", lw=1, label="ρ=0.75 threshold")
    ax.axvline(0.50, color="gray", linestyle=":",  lw=1, label="ρ=0.50 threshold")
    ax.set_xlabel("Spearman ρ  (lower = greater attentional shift)")
    ax.set_title("SHAP Rank Correlation: Baseline vs Cost-Sensitive", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(f"{FIGURE_DIR}/spearman_correlation.png", dpi=150, bbox_inches="tight")
    plt.show()
 
    return df


  
def plot_ablation(ablation_results: dict, save=True):
    """
    Plot U2R recall and TEL across Conservative/Moderate/Aggressive regimes.
 
    ablation_results: {
        regime: {"tel": float, "u2r_recall": float, "r2l_recall": float}
    }
    """
    import os; os.makedirs(FIGURE_DIR, exist_ok=True)
    regimes    = list(ablation_results.keys())
    tel        = [ablation_results[r]["tel"]        for r in regimes]
    u2r_recall = [ablation_results[r]["u2r_recall"] for r in regimes]
    r2l_recall = [ablation_results[r]["r2l_recall"] for r in regimes]
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
 
    ax1.plot(regimes, tel, "o-", color="#F44336", lw=2)
    ax1.set_title("TEL vs Cost Regime", fontweight="bold")
    ax1.set_ylabel("Total Economic Loss (USD)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.grid(alpha=0.3)
 
    ax2.plot(regimes, u2r_recall, "o-", color="#F50057", lw=2, label="U2R recall")
    ax2.plot(regimes, r2l_recall, "s-", color="#9C27B0", lw=2, label="R2L recall")
    ax2.set_title("Minority Class Recall vs Cost Regime", fontweight="bold")
    ax2.set_ylabel("Recall")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)
 
    plt.suptitle("Ablation Study: Cost Regime Impact", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(f"{FIGURE_DIR}/ablation_study.png", dpi=150, bbox_inches="tight")
    plt.show()