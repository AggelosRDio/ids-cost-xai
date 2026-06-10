"""
Xai.py
------
Explainability suite for the cost-sensitive IDS pipeline.

Responsibilities:
  - SHAP: global rankings, per-class rankings, local waterfall plots
  - LIME: local explanations for high-cost minority-class samples
  - XGBoost native importance: baseline vs cost-sensitive comparison
  - XAI method agreement: SHAP / LIME / XGB native side-by-side
  - Spearman rank correlation: quantify attentional shift across regimes
  - Security feature shift: track critical features across regimes

NOTE on TreeExplainer vs calibrated model:
  SHAP operates on the BASE (uncalibrated) tree model to access tree
  structure. Probabilities shown in local plots come from the calibrated
  model. This is intentional — document the distinction in the paper.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

FIGURE_DIR = "results/xai"
os.makedirs(FIGURE_DIR, exist_ok=True)

CLASS_COLORS = {
    "Normal": "#4CAF50", "DoS": "#F44336",
    "Probe": "#FF9800", "R2L": "#9C27B0", "U2R": "#F50057",
}

DEFAULT_SECURITY_FEATURES = (
    "root_shell", "su_attempted", "num_failed_logins",
    "num_shells", "num_access_files", "logged_in",
    "num_compromised", "num_root", "num_file_creations",
)


# ==================================================
# UTILITIES
# ==================================================

def ensure_dataframe(X, feature_names):
    """Return X as a DataFrame with named columns."""
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X, columns=feature_names)


def _predict_encoded(model, X):
    """Argmax prediction from predict_proba."""
    return np.argmax(model.predict_proba(X), axis=1)


# ==================================================
# SHAP — SAMPLING
# ==================================================

def make_shap_sample(X, y, feature_names, n_per_class=200, random_state=42):
    """
    Stratified sample for SHAP global analysis.
    Random sampling under-represents rare classes (U2R). This samples
    up to n_per_class instances per class so minority-class SHAP values
    are reliable.
    Returns (X_sample DataFrame, y_sample array).
    """
    X_df  = ensure_dataframe(X, feature_names)
    y_arr = np.asarray(y)

    rng = np.random.RandomState(random_state)
    selected_idx = []

    for cls in np.unique(y_arr):
        cls_idx = np.where(y_arr == cls)[0]
        n       = min(n_per_class, len(cls_idx))
        selected_idx.extend(rng.choice(cls_idx, size=n, replace=False))

    selected_idx = np.array(selected_idx)
    rng.shuffle(selected_idx)

    X_sample = X_df.iloc[selected_idx].reset_index(drop=True)
    y_sample = y_arr[selected_idx]

    print(f"[XAI] Stratified SHAP sample: {len(X_sample)} rows")
    for cls, count in zip(*np.unique(y_sample, return_counts=True)):
        print(f"  class {cls}: {count}")

    return X_sample, y_sample


# ==================================================
# SHAP — COMPUTATION
# ==================================================

def get_shap_values(model, X_sample, feature_names):
    """
    Compute SHAP values via TreeExplainer.
    Returns shap_values (n_samples, n_features, n_classes) and explainer.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=2)
    elif shap_values.ndim == 2:
        shap_values = shap_values[:, :, np.newaxis]

    return shap_values, explainer


def compute_shap_ranking(model, X_sample, feature_names, model_name):
    """
    Compute global SHAP feature ranking (mean |SHAP|) for one model.
    Returns (shap_values array, ranking DataFrame).
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        importance = np.mean(
            [np.abs(v).mean(axis=0) for v in shap_values], axis=0
        )
    elif shap_values.ndim == 3:
        importance = np.abs(shap_values).mean(axis=(0, 2))
    else:
        importance = np.abs(shap_values).mean(axis=0)

    ranking = (
        pd.DataFrame({"feature": feature_names,
                      f"{model_name}_importance": importance})
        .sort_values(f"{model_name}_importance", ascending=False)
        .reset_index(drop=True)
    )
    ranking[f"{model_name}_rank"] = ranking.index + 1

    # Normalise shap_values to (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=2)
    elif shap_values.ndim == 2:
        shap_values = shap_values[:, :, np.newaxis]

    return shap_values, ranking


def compute_all_shap_rankings(raw_models, X_sample, feature_names):
    """Compute SHAP rankings for all models in raw_models dict."""
    shap_values_dict, rankings = {}, {}
    for model_name, model in raw_models.items():
        print(f"[XAI] Computing SHAP for: {model_name}")
        shap_values_dict[model_name], rankings[model_name] = compute_shap_ranking(
            model, X_sample, feature_names, model_name
        )
    return shap_values_dict, rankings


def compute_class_specific_ranking(shap_values, feature_names, class_idx, model_name, class_name):
    """SHAP feature ranking for one target class."""
    if isinstance(shap_values, list):
        class_shap = shap_values[class_idx]
    elif shap_values.ndim == 3:
        class_shap = shap_values[:, :, class_idx]
    else:
        raise ValueError("Class-specific ranking requires multiclass SHAP values.")

    importance = np.abs(class_shap).mean(axis=0)
    col_imp    = f"{model_name}_{class_name}_importance"
    col_rank   = f"{model_name}_{class_name}_rank"

    ranking = (
        pd.DataFrame({"feature": feature_names, col_imp: importance})
        .sort_values(col_imp, ascending=False)
        .reset_index(drop=True)
    )
    ranking[col_rank] = ranking.index + 1
    return ranking


def compute_class_specific_rankings(shap_values_dict, feature_names, encoder,
                                     target_classes=("R2L", "U2R")):
    """Class-specific SHAP rankings for each target class across all models."""
    class_to_idx = {label: idx for idx, label in enumerate(encoder.classes_)}
    out = {}
    for cls in target_classes:
        cls_idx = class_to_idx[cls]
        out[cls] = {}
        for model_name, shap_values in shap_values_dict.items():
            out[cls][model_name] = compute_class_specific_ranking(
                shap_values, feature_names, cls_idx, model_name, cls
            )
    return out


# ==================================================
# SPEARMAN CORRELATION
# ==================================================

def get_rank_series(ranking, rank_col):
    return ranking.set_index("feature")[rank_col].sort_index()


def compute_global_spearman(rankings, regimes):
    """Spearman ρ between baseline and each cost-sensitive regime (global)."""
    baseline_ranks = get_rank_series(rankings["baseline"], "baseline_rank")
    rows = []
    for regime in regimes:
        regime_ranks = get_rank_series(rankings[regime], f"{regime}_rank")
        rho, p = spearmanr(baseline_ranks, regime_ranks)
        rows.append({"class": "Global",
                     "comparison": f"baseline_vs_{regime}",
                     "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


def compute_class_specific_spearman(class_specific_rankings, regimes,
                                     target_classes=("R2L", "U2R")):
    """Spearman ρ per high-cost class."""
    rows = []
    for cls in target_classes:
        base_col   = f"baseline_{cls}_rank"
        base_ranks = class_specific_rankings[cls]["baseline"].set_index("feature")[base_col].sort_index()
        for regime in regimes:
            regime_col   = f"{regime}_{cls}_rank"
            regime_ranks = class_specific_rankings[cls][regime].set_index("feature")[regime_col].sort_index()
            rho, p = spearmanr(base_ranks, regime_ranks)
            rows.append({"class": cls,
                         "comparison": f"baseline_vs_{regime}",
                         "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


# ==================================================
# SECURITY FEATURE SHIFT
# ==================================================

def compute_security_feature_shift(rankings, security_features=None):
    """Track rank and importance changes for security-critical features."""
    if security_features is None:
        security_features = DEFAULT_SECURITY_FEATURES

    available = set(rankings["baseline"]["feature"].values)
    missing   = [f for f in security_features if f not in available]
    if missing:
        print(f"[WARNING] Security features not found: {missing}")

    rows = []
    for feat in security_features:
        row = {"feature": feat}
        for model_name, ranking in rankings.items():
            imp_col  = f"{model_name}_importance"
            rank_col = f"{model_name}_rank"
            match = ranking[ranking["feature"] == feat]
            row[rank_col] = int(match[rank_col].iloc[0])  if len(match) else None
            row[imp_col]  = float(match[imp_col].iloc[0]) if len(match) else None
        rows.append(row)

    shift_df = pd.DataFrame(rows)
    for regime in [n for n in rankings if n != "baseline"]:
        shift_df[f"{regime}_rank_change"] = (
            shift_df["baseline_rank"] - shift_df[f"{regime}_rank"]
        )
        shift_df[f"{regime}_importance_change"] = (
            shift_df[f"{regime}_importance"] - shift_df["baseline_importance"]
        )
    return shift_df


# ==================================================
# SHAP — GLOBAL PLOTS
# ==================================================

def plot_shap_global(shap_values, X_human, feature_names, le, save=True):
    """Beeswarm and bar summary plots for each class."""
    n_classes = shap_values.shape[2]
    for cls_idx in range(n_classes):
        cls_name = le.inverse_transform([cls_idx])[0]
        vals     = shap_values[:, :, cls_idx]

        fig, ax = plt.subplots(figsize=(11, 7))
        shap.summary_plot(vals, X_human, feature_names=feature_names,
                          show=False, max_display=15)
        plt.title(f"SHAP Summary — {cls_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save:
            plt.savefig(f"{FIGURE_DIR}/plots/shap_beeswarm_{cls_name}.png",
                        dpi=150, bbox_inches="tight")
        plt.show()

        fig, ax = plt.subplots(figsize=(11, 6))
        shap.summary_plot(vals, X_human, feature_names=feature_names,
                          plot_type="bar", show=False, max_display=15)
        plt.title(f"SHAP Feature Importance — {cls_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save:
            plt.savefig(f"{FIGURE_DIR}/plots/shap_bar_{cls_name}.png",
                        dpi=150, bbox_inches="tight")
        plt.show()


def save_shap_summary_plots(rankings, output_dir=None, max_display=15):
    """Bar chart of mean absolute SHAP importance per model."""
    output_dir = output_dir or f"{FIGURE_DIR}/plots"
    os.makedirs(output_dir, exist_ok=True)

    for model_name, ranking in rankings.items():
        imp_col  = f"{model_name}_importance"
        plot_df  = ranking.head(max_display).iloc[::-1]

        plt.figure(figsize=(8, 6))
        plt.barh(plot_df["feature"], plot_df[imp_col])
        plt.xlabel("Mean absolute SHAP value")
        plt.title(f"SHAP Feature Importance — {model_name}")
        plt.tight_layout()

        path = os.path.join(output_dir, f"shap_summary_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[XAI] Saved {path}")


# ==================================================
# SHAP — LOCAL (WATERFALL)
# ==================================================

def plot_shap_local(
    sample_idx,
    X_test, X_test_human,
    y_test_enc, y_simple_pred_enc, y_cost_pred_enc, y_probs,
    explainer, shap_values,
    feature_names, le, cost_matrix,
    save=True,
):
    """
    Waterfall plot for a single sample.

    IMPORTANT: sample_idx must be a positional index into shap_values
    (i.e. within the SHAP sample, not the full test set). Use
    find_corrected_samples_in_shap_sample() to get safe indices.
    """
    true_enc   = y_test_enc[sample_idx]
    simple_enc = y_simple_pred_enc[sample_idx]
    cost_enc   = y_cost_pred_enc[sample_idx]

    true_name   = le.inverse_transform([true_enc])[0]
    simple_name = le.inverse_transform([simple_enc])[0]
    cost_name   = le.inverse_transform([cost_enc])[0]

    print(f"\nSample {sample_idx}")
    print(f"  True:              {true_name}")
    print(f"  Baseline:          {simple_name}")
    print(f"  Cost-sensitive:    {cost_name}")

    print("\n  Class probabilities:")
    for i, prob in enumerate(y_probs[sample_idx]):
        print(f"    {le.inverse_transform([i])[0]:<12} {prob:.4f}")

    print("\n  Expected cost per decision:")
    for c in range(len(cost_matrix)):
        ec = np.sum(y_probs[sample_idx] * cost_matrix[c])
        print(f"    {le.inverse_transform([c])[0]:<12} {ec:.4f}")

    sv = shap_values[sample_idx, :, true_enc]
    base_val = (explainer.expected_value[true_enc]
                if hasattr(explainer.expected_value, "__len__")
                else explainer.expected_value)

    exp = shap.Explanation(
        values=sv,
        base_values=base_val,
        data=X_test_human.iloc[sample_idx].values,
        feature_names=feature_names,
    )
    fig = plt.figure(figsize=(13, 8))
    shap.waterfall_plot(exp, max_display=12, show=False)
    plt.title(
        f"SHAP Waterfall — {true_name}\n"
        f"baseline: {simple_name}  →  cost-sensitive: {cost_name}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if save:
        os.makedirs(f"{FIGURE_DIR}/plots", exist_ok=True)
        plt.savefig(f"{FIGURE_DIR}/plots/shap_waterfall_{true_name}_{sample_idx}.png",
                    dpi=150, bbox_inches="tight")
    plt.show()


# ==================================================
# SAMPLE SELECTION HELPERS
# ==================================================

def find_corrected_samples(y_test_enc, y_simple_pred_enc, y_cost_pred_enc, target_class_idx):
    """Indices (in full test set) where cost-sensitive corrected a baseline error."""
    return np.where(
        (y_simple_pred_enc != y_test_enc) &
        (y_cost_pred_enc   == y_test_enc) &
        (y_test_enc        == target_class_idx)
    )[0]


def find_corrected_samples_in_shap_sample(
    y_test_enc, y_simple_pred_enc, y_cost_pred_enc,
    target_class_idx, shap_n_rows,
):
    """
    Same as find_corrected_samples but restricted to indices that exist
    in the SHAP sample (indices < shap_n_rows).
    Use this when you need to index into shap_values directly.
    """
    all_corrected = find_corrected_samples(
        y_test_enc, y_simple_pred_enc, y_cost_pred_enc, target_class_idx
    )
    return all_corrected[all_corrected < shap_n_rows]


def find_lime_candidate_samples(
    raw_models, X_test, y_test, feature_names, encoder,
    target_model_name="aggressive",
    target_classes=("R2L", "U2R"),
    max_samples_per_class=2,
):
    """
    Find interesting samples for LIME: prioritise cases where baseline
    is wrong but the cost-sensitive model is correct.
    """
    if "baseline" not in raw_models:
        raise ValueError("raw_models must include a 'baseline' model.")
    if target_model_name not in raw_models:
        raise ValueError(f"raw_models missing: {target_model_name}")

    X_df  = ensure_dataframe(X_test, feature_names)
    y_arr = np.asarray(y_test)

    baseline_pred = _predict_encoded(raw_models["baseline"],          X_df)
    target_pred   = _predict_encoded(raw_models[target_model_name],   X_df)

    rows = []
    for cls in target_classes:
        cls_idx = int(np.where(encoder.classes_ == cls)[0][0])
        cls_indices = np.where(y_arr == cls_idx)[0]

        corrected = [i for i in cls_indices
                     if baseline_pred[i] != y_arr[i] and target_pred[i] == y_arr[i]]
        fallback  = [i for i in cls_indices
                     if target_pred[i] == y_arr[i] and i not in corrected]

        selected = corrected[:max_samples_per_class]
        if len(selected) < max_samples_per_class:
            selected.extend(fallback[:max_samples_per_class - len(selected)])

        for idx in selected:
            rows.append({
                "sample_index":            int(idx),
                "true_class":              encoder.inverse_transform([y_arr[idx]])[0],
                "baseline_prediction":     encoder.inverse_transform([baseline_pred[idx]])[0],
                f"{target_model_name}_prediction":
                                           encoder.inverse_transform([target_pred[idx]])[0],
                "selection_reason": (
                    "baseline_wrong_target_correct" if idx in corrected
                    else "target_correct_fallback"
                ),
            })
    return pd.DataFrame(rows)


# ==================================================
# LIME
# ==================================================

def build_lime_explainer(X_train, feature_names, class_names):
    """Build a LIME tabular explainer fit on the training distribution."""
    return lime.lime_tabular.LimeTabularExplainer(
        np.asarray(X_train),
        feature_names=list(feature_names),
        class_names=list(class_names),
        mode="classification",
        random_state=42,
    )


def plot_lime_local(lime_explainer, cal_model, sample_idx,
                    X_test, X_test_human, y_test_enc, le,
                    feature_names, top_n=12, save=True):
    """LIME explanation for a single sample."""
    true_enc  = y_test_enc[sample_idx]
    true_name = le.inverse_transform([true_enc])[0]

    exp = lime_explainer.explain_instance(
        np.asarray(X_test)[sample_idx],
        cal_model.predict_proba,
        num_features=top_n,
        labels=list(range(len(le.classes_))),
    )
    fig = exp.as_pyplot_figure(label=true_enc)
    fig.set_size_inches(11, 6)
    plt.title(f"LIME — {true_name} (sample {sample_idx})",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        os.makedirs(f"{FIGURE_DIR}/plots", exist_ok=True)
        plt.savefig(f"{FIGURE_DIR}/plots/lime_{true_name}_{sample_idx}.png",
                    dpi=150, bbox_inches="tight")
    plt.show()
    return exp


def run_lime_analysis(
    raw_models, X_train, X_test, y_test, feature_names, encoder,
    target_model_name="aggressive",
    target_classes=("R2L", "U2R"),
    max_samples_per_class=2,
    num_features=10,
    output_dir=None,
    random_state=42,
):
    """Full LIME pipeline: find candidates → explain → save HTML + index CSV."""
    output_dir = output_dir or f"{FIGURE_DIR}/lime"
    os.makedirs(output_dir, exist_ok=True)

    X_train_df = ensure_dataframe(X_train, feature_names)
    X_test_df  = ensure_dataframe(X_test,  feature_names)

    candidates_df = find_lime_candidate_samples(
        raw_models=raw_models, X_test=X_test_df, y_test=y_test,
        feature_names=feature_names, encoder=encoder,
        target_model_name=target_model_name,
        target_classes=target_classes,
        max_samples_per_class=max_samples_per_class,
    )
    candidates_df.to_csv(os.path.join(output_dir, "lime_candidates.csv"), index=False)
    print(f"[LIME] Candidates: {len(candidates_df)} samples")

    if candidates_df.empty:
        print("[LIME] No candidates found.")
        return {"lime_candidates": candidates_df, "lime_output_dir": output_dir}

    explainer = LimeTabularExplainer(
        training_data=X_train_df.values,
        feature_names=list(feature_names),
        class_names=list(encoder.classes_),
        mode="classification",
        discretize_continuous=True,
        random_state=random_state,
    )

    explanation_rows = []
    for _, row in candidates_df.iterrows():
        sample_idx = int(row["sample_index"])
        true_class = row["true_class"]
        instance   = X_test_df.iloc[sample_idx].values

        for model_name in ["baseline", target_model_name]:
            model      = raw_models[model_name]
            pred_enc   = _predict_encoded(model, X_test_df.iloc[[sample_idx]])[0]
            pred_class = encoder.inverse_transform([pred_enc])[0]

            explanation = explainer.explain_instance(
                data_row=instance,
                predict_fn=model.predict_proba,
                labels=[pred_enc],
                num_features=num_features,
            )
            fname = (f"lime_{model_name}_s{sample_idx}"
                     f"_true{true_class}_pred{pred_class}.html")
            fpath = os.path.join(output_dir, fname)
            explanation.save_to_file(fpath)
            explanation_rows.append({
                "sample_index": sample_idx, "model": model_name,
                "true_class": true_class, "predicted_class": pred_class,
                "explanation_file": fpath,
            })
            print(f"[LIME] Saved {fpath}")

    explanations_df = pd.DataFrame(explanation_rows)
    explanations_df.to_csv(os.path.join(output_dir, "lime_index.csv"), index=False)
    return {"lime_candidates": candidates_df,
            "lime_explanations": explanations_df,
            "lime_output_dir": output_dir}


# ==================================================
# XGBoost NATIVE IMPORTANCE
# ==================================================

def plot_xgb_native_importance(baseline_model, costsens_model,
                                feature_names, top_n=20, save=True):
    """Side-by-side XGBoost native importance (gain / cover / weight)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, imp_type in zip(axes, ["gain", "cover", "weight"]):
        base_scores = baseline_model.get_booster().get_score(importance_type=imp_type)
        cost_scores = costsens_model.get_booster().get_score(importance_type=imp_type)

        all_feats = sorted(set(base_scores) | set(cost_scores))
        base_vals = [base_scores.get(f, 0) for f in all_feats]
        cost_vals = [cost_scores.get(f, 0) for f in all_feats]

        order = np.argsort(cost_vals)[-top_n:]
        feats = [all_feats[i] for i in order]
        bv    = [base_vals[i] for i in order]
        cv    = [cost_vals[i] for i in order]

        y = np.arange(len(feats))
        ax.barh(y - 0.2, bv, 0.4, label="Baseline",       color="#1976D2", alpha=0.8)
        ax.barh(y + 0.2, cv, 0.4, label="Cost-Sensitive", color="#F44336", alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(feats, fontsize=8)
        ax.set_title(f"XGB Native: {imp_type.capitalize()}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("XGBoost Native Feature Importance: Baseline vs Cost-Sensitive",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        os.makedirs(f"{FIGURE_DIR}/plots", exist_ok=True)
        plt.savefig(f"{FIGURE_DIR}/plots/xgb_native_importance.png",
                    dpi=150, bbox_inches="tight")
    plt.show()


# ==================================================
# XAI METHOD AGREEMENT
# ==================================================

def plot_xai_agreement(shap_values, lime_exp_list, baseline_model,
                        feature_names, le, target_class_idx,
                        top_n=10, save=True):
    cls_name = le.inverse_transform([target_class_idx])[0]

    # SHAP: class-specific mean |value|
    shap_imp  = np.abs(shap_values[:, :, target_class_idx]).mean(axis=0)
    shap_rank = pd.Series(shap_imp, index=feature_names).nlargest(top_n)

    # XGB native gain — global (not class-specific, by design)
    gain_scores = baseline_model.get_booster().get_score(importance_type="gain")
    gain_series = pd.Series(gain_scores).reindex(feature_names).fillna(0)

    # Align: show gain only for the top-N features SHAP selected
    # This makes the comparison honest — same feature set across all three panels
    top_features = shap_rank.index.tolist()
    gain_aligned = gain_series.reindex(top_features).fillna(0).sort_values()

    # LIME: aggregate |weights| across explanations
    lime_agg = {}
    for exp in lime_exp_list:
        for feat, weight in exp.as_list(label=target_class_idx):
            base = feat.split(" ")[0].split(">")[0].split("<")[0].strip()
            lime_agg[base] = lime_agg.get(base, 0) + abs(weight)
    lime_series = pd.Series(lime_agg).reindex(top_features).fillna(0).sort_values()

    shap_plot = shap_rank.reindex(top_features).fillna(0).sort_values()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, series, title in zip(
        axes,
        [shap_plot, gain_aligned, lime_series],
        ["SHAP (mean |value|)",
         "XGB Native (gain)\n[global — not class-specific]",
         "LIME (agg. |weight|)"],
    ):
        series.plot.barh(
            ax=ax,
            color=CLASS_COLORS.get(cls_name, "#607D8B"),
            alpha=0.85, edgecolor="white",
        )
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(f"XAI Method Agreement — {cls_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        os.makedirs(f"{FIGURE_DIR}/plots", exist_ok=True)
        plt.savefig(f"{FIGURE_DIR}/plots/xai_agreement_{cls_name}.png",
                    dpi=150, bbox_inches="tight")
    plt.show()

# ==================================================
# PERSISTENCE
# ==================================================

def save_xai_tables(spearman_summary_df, security_shift_df, output_dir=None):
    output_dir = output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    spearman_summary_df.to_csv(os.path.join(output_dir, "spearman_summary.csv"),   index=False)
    security_shift_df.to_csv( os.path.join(output_dir, "security_feature_shift.csv"), index=False)
    print(f"[XAI] Tables saved to {output_dir}")


# ==================================================
# ORCHESTRATOR
# ==================================================

def run_xai_analysis(
    raw_models, X_test, y_test, feature_names, encoder,
    regimes=("conservative", "moderate", "aggressive"),
    n_per_class=200,
    random_state=42,
    output_dir=None,
    target_classes=("R2L", "U2R"),
    security_features=None,
):
    """
    End-to-end XAI audit. Receives already-trained raw (uncalibrated) models.

    Expected raw_models keys: "baseline", "conservative", "moderate", "aggressive".

    Steps:
      1. Stratified SHAP sample
      2. Global SHAP rankings + Spearman correlation
      3. Security-critical feature shift
      4. Class-specific rankings (R2L, U2R) + Spearman
      5. Save tables and bar-chart plots
    """
    output_dir = output_dir or FIGURE_DIR

    if "baseline" not in raw_models:
        raise ValueError("raw_models must include 'baseline'.")
    missing = [r for r in regimes if r not in raw_models]
    if missing:
        raise ValueError(f"raw_models missing: {missing}")

    X_sample, y_sample = make_shap_sample(
        X_test, y_test, feature_names,
        n_per_class=n_per_class, random_state=random_state,
    )
    shap_values_dict, rankings = compute_all_shap_rankings(raw_models, X_sample, feature_names)

    global_spearman_df    = compute_global_spearman(rankings, regimes)
    security_shift_df     = compute_security_feature_shift(rankings, security_features)
    class_specific_ranks  = compute_class_specific_rankings(
        shap_values_dict, feature_names, encoder, target_classes
    )
    class_spearman_df     = compute_class_specific_spearman(
        class_specific_ranks, regimes, target_classes
    )
    spearman_summary_df   = pd.concat([global_spearman_df, class_spearman_df],
                                       ignore_index=True)

    save_xai_tables(spearman_summary_df, security_shift_df, output_dir)
    save_shap_summary_plots(rankings, output_dir=os.path.join(output_dir, "plots"))

    return {
        "X_sample":               X_sample,
        "y_sample":               y_sample,
        "shap_values_dict":       shap_values_dict,
        "rankings":               rankings,
        "global_spearman_df":     global_spearman_df,
        "class_specific_rankings":class_specific_ranks,
        "class_spearman_df":      class_spearman_df,
        "spearman_summary_df":    spearman_summary_df,
        "security_shift_df":      security_shift_df,
    }
