import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from scipy.stats import spearmanr


DEFAULT_SECURITY_FEATURES = (
    "root_shell",
    "su_attempted",
    "num_failed_logins",
    "num_shells",
    "num_access_files",
    "logged_in",
    "num_compromised",
    "num_root",
    "num_file_creations",
)


def ensure_dataframe(X, feature_names):
    """
    Ensure that the feature matrix is a pandas DataFrame with readable feature names.
    This is useful because SHAP plots and rankings are more interpretable with named columns.
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()

    return pd.DataFrame(X, columns=feature_names)


def make_shap_sample(X, y, feature_names, n_per_class=200, random_state=42):
    """
    Create a stratified fixed sample for SHAP analysis.

    Random sampling can under-represent rare classes such as U2R.
    This function samples up to n_per_class instances from each class,
    so class-specific SHAP analysis for minority classes is more reliable.
    """
    X_df = ensure_dataframe(X, feature_names)
    y_arr = np.asarray(y)

    if len(X_df) != len(y_arr):
        raise ValueError("X and y must have the same number of rows for stratified SHAP sampling.")

    rng = np.random.RandomState(random_state)
    selected_idx = []

    for cls in np.unique(y_arr):
        cls_idx = np.where(y_arr == cls)[0]
        n = min(n_per_class, len(cls_idx))
        sampled_idx = rng.choice(cls_idx, size=n, replace=False)
        selected_idx.extend(sampled_idx)

    selected_idx = np.array(selected_idx)
    rng.shuffle(selected_idx)

    X_sample = X_df.iloc[selected_idx].reset_index(drop=True)
    y_sample = y_arr[selected_idx]

    print(f"[XAI] Stratified SHAP sample size: {len(X_sample)}")
    for cls, count in zip(*np.unique(y_sample, return_counts=True)):
        print(f"[XAI] Class {cls}: {count} samples")

    return X_sample, y_sample


def compute_shap_ranking(model, X_sample, feature_names, model_name):
    """
    Compute SHAP values and a global feature ranking for a model.

    For multiclass classification, SHAP values usually have shape:
    (samples, features, classes).

    We compute global importance as the mean absolute SHAP value across
    samples and classes.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        # Older SHAP versions may return a list with one array per class.
        importance = np.mean(
            [np.abs(class_values).mean(axis=0) for class_values in shap_values],
            axis=0,
        )
    elif len(shap_values.shape) == 3:
        # Newer SHAP versions may return shape: (samples, features, classes).
        importance = np.abs(shap_values).mean(axis=(0, 2))
    else:
        # Binary or single-output case.
        importance = np.abs(shap_values).mean(axis=0)

    ranking = pd.DataFrame(
        {
            "feature": feature_names,
            f"{model_name}_importance": importance,
        }
    ).sort_values(
        f"{model_name}_importance",
        ascending=False,
    ).reset_index(drop=True)

    ranking[f"{model_name}_rank"] = ranking.index + 1

    return shap_values, ranking


def compute_all_shap_rankings(raw_models, X_sample, feature_names):
    """
    Compute SHAP values and global rankings for all provided raw models.

    raw_models should contain already trained models, for example:
    {
        "baseline": baseline_xgb_for_xai,
        "conservative": conservative_xgb_for_xai,
        "moderate": moderate_xgb_for_xai,
        "aggressive": aggressive_xgb_for_xai,
    }
    """
    shap_values_dict = {}
    rankings = {}

    for model_name, model in raw_models.items():
        print(f"[XAI] Computing SHAP ranking for: {model_name}")
        shap_values_dict[model_name], rankings[model_name] = compute_shap_ranking(
            model,
            X_sample,
            feature_names,
            model_name,
        )

    return shap_values_dict, rankings


def get_rank_series(ranking, rank_col):
    """
    Convert a feature ranking DataFrame into a feature-indexed rank Series.
    """
    return ranking.set_index("feature")[rank_col].sort_index()


def compute_global_spearman(rankings, regimes):
    """
    Compute Spearman rank correlation between baseline and each cost-sensitive regime.
    """
    baseline_ranks = get_rank_series(rankings["baseline"], "baseline_rank")

    rows = []

    for regime in regimes:
        regime_ranks = get_rank_series(rankings[regime], f"{regime}_rank")

        rho, p_value = spearmanr(baseline_ranks, regime_ranks)

        rows.append(
            {
                "class": "Global",
                "comparison": f"baseline_vs_{regime}",
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(rows)


def compute_security_feature_shift(rankings, security_features=None):
    """
    Track how selected security-critical features change in rank and SHAP importance.
    """
    if security_features is None:
        security_features = DEFAULT_SECURITY_FEATURES

    available_features = set(rankings["baseline"]["feature"].values)
    missing = [f for f in security_features if f not in available_features]

    if missing:
        print(f"[WARNING] Security features not found in rankings: {missing}")

    rows = []

    for feature in security_features:
        row = {"feature": feature}

        for model_name, ranking in rankings.items():
            rank_col = f"{model_name}_rank"
            importance_col = f"{model_name}_importance"

            match = ranking[ranking["feature"] == feature]

            if len(match) > 0:
                row[f"{model_name}_rank"] = int(match[rank_col].iloc[0])
                row[f"{model_name}_importance"] = float(match[importance_col].iloc[0])
            else:
                row[f"{model_name}_rank"] = None
                row[f"{model_name}_importance"] = None

        rows.append(row)

    shift_df = pd.DataFrame(rows)

    for regime in [name for name in rankings.keys() if name != "baseline"]:
        shift_df[f"{regime}_rank_change"] = (
            shift_df["baseline_rank"] - shift_df[f"{regime}_rank"]
        )

        shift_df[f"{regime}_importance_change"] = (
            shift_df[f"{regime}_importance"] - shift_df["baseline_importance"]
        )

    return shift_df


def compute_class_specific_ranking(
    shap_values,
    feature_names,
    class_idx,
    model_name,
    class_name,
):
    """
    Compute SHAP feature ranking for one target class.

    This is important because global SHAP rankings may hide changes
    in high-cost minority classes such as R2L and U2R.
    """
    if isinstance(shap_values, list):
        class_shap = shap_values[class_idx]
    elif len(shap_values.shape) == 3:
        class_shap = shap_values[:, :, class_idx]
    else:
        raise ValueError("Class-specific SHAP ranking requires multiclass SHAP values.")

    importance = np.abs(class_shap).mean(axis=0)

    ranking = pd.DataFrame(
        {
            "feature": feature_names,
            f"{model_name}_{class_name}_importance": importance,
        }
    ).sort_values(
        f"{model_name}_{class_name}_importance",
        ascending=False,
    ).reset_index(drop=True)

    ranking[f"{model_name}_{class_name}_rank"] = ranking.index + 1

    return ranking


def compute_class_specific_rankings(
    shap_values_dict,
    feature_names,
    encoder,
    target_classes=("R2L", "U2R"),
):
    """
    Compute class-specific SHAP rankings for selected classes.
    """
    class_to_idx = {label: idx for idx, label in enumerate(encoder.classes_)}

    class_specific_rankings = {}

    for target_class in target_classes:
        class_idx = class_to_idx[target_class]
        class_specific_rankings[target_class] = {}

        for model_name, shap_values in shap_values_dict.items():
            class_specific_rankings[target_class][model_name] = compute_class_specific_ranking(
                shap_values,
                feature_names,
                class_idx,
                model_name,
                target_class,
            )

    return class_specific_rankings


def compute_class_specific_spearman(
    class_specific_rankings,
    regimes,
    target_classes=("R2L", "U2R"),
):
    """
    Compute Spearman rank correlation for each high-cost target class.
    """
    rows = []

    for target_class in target_classes:
        baseline_rank_col = f"baseline_{target_class}_rank"

        baseline_ranks = class_specific_rankings[target_class]["baseline"].set_index(
            "feature"
        )[baseline_rank_col].sort_index()

        for regime in regimes:
            regime_rank_col = f"{regime}_{target_class}_rank"

            regime_ranks = class_specific_rankings[target_class][regime].set_index(
                "feature"
            )[regime_rank_col].sort_index()

            rho, p_value = spearmanr(baseline_ranks, regime_ranks)

            rows.append(
                {
                    "class": target_class,
                    "comparison": f"baseline_vs_{regime}",
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(rows)


def save_xai_tables(
    spearman_summary_df,
    security_shift_df,
    output_dir="results/xai",
):
    """
    Save XAI result tables as CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    spearman_path = os.path.join(output_dir, "spearman_summary.csv")
    security_path = os.path.join(output_dir, "security_feature_shift.csv")

    spearman_summary_df.to_csv(spearman_path, index=False)
    security_shift_df.to_csv(security_path, index=False)

    print(f"[XAI] Saved {spearman_path}")
    print(f"[XAI] Saved {security_path}")


def save_shap_summary_plots(
    rankings,
    output_dir="results/xai/plots",
    max_display=15,
):
    """
    Save SHAP mean absolute importance plots for all models.

    These plots summarize the most important features according to
    mean absolute SHAP values.
    """
    os.makedirs(output_dir, exist_ok=True)

    for model_name, ranking in rankings.items():
        importance_col = f"{model_name}_importance"

        plot_df = ranking.head(max_display).iloc[::-1]

        plt.figure(figsize=(8, 6))
        plt.barh(plot_df["feature"], plot_df[importance_col])
        plt.xlabel("Mean absolute SHAP value")
        plt.ylabel("Feature")
        plt.title(f"SHAP Feature Importance - {model_name}")
        plt.tight_layout()

        path = os.path.join(output_dir, f"shap_summary_{model_name}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[XAI] Saved {path}")


def run_xai_analysis(
    raw_models,
    X_test,
    y_test,
    feature_names,
    encoder,
    regimes=("conservative", "moderate", "aggressive"),
    random_state=42,
    n_per_class=200,
    output_dir="results/xai",
    target_classes=("R2L", "U2R"),
    security_features=None,
):
    """
    End-to-end XAI audit for the cost-sensitive IDS pipeline.

    This function does not train models. It receives already trained raw models
    from the modelling pipeline and explains them using SHAP.

    Expected raw_models format:
    {
        "baseline": baseline_xgb_for_xai,
        "conservative": conservative_xgb_for_xai,
        "moderate": moderate_xgb_for_xai,
        "aggressive": aggressive_xgb_for_xai,
    }

    This function:
    1. Creates a stratified SHAP sample using y_test.
    2. Computes global SHAP rankings.
    3. Computes global Spearman rank correlations as a sanity check.
    4. Tracks security-critical feature shifts.
    5. Computes class-specific SHAP rankings for R2L and U2R.
    6. Computes class-specific Spearman correlations.
    7. Saves result tables and plots.
    """
    if "baseline" not in raw_models:
        raise ValueError("raw_models must include a 'baseline' model.")

    missing_regimes = [regime for regime in regimes if regime not in raw_models]
    if missing_regimes:
        raise ValueError(f"raw_models is missing cost-sensitive models: {missing_regimes}")

    X_sample, y_sample = make_shap_sample(
        X_test,
        y_test,
        feature_names,
        n_per_class=n_per_class,
        random_state=random_state,
    )

    shap_values_dict, rankings = compute_all_shap_rankings(
        raw_models=raw_models,
        X_sample=X_sample,
        feature_names=feature_names,
    )

    global_spearman_df = compute_global_spearman(rankings, regimes)

    security_shift_df = compute_security_feature_shift(
        rankings,
        security_features=security_features,
    )

    class_specific_rankings = compute_class_specific_rankings(
        shap_values_dict=shap_values_dict,
        feature_names=feature_names,
        encoder=encoder,
        target_classes=target_classes,
    )

    class_spearman_df = compute_class_specific_spearman(
        class_specific_rankings=class_specific_rankings,
        regimes=regimes,
        target_classes=target_classes,
    )

    spearman_summary_df = pd.concat(
        [global_spearman_df, class_spearman_df],
        ignore_index=True,
    )

    save_xai_tables(
        spearman_summary_df=spearman_summary_df,
        security_shift_df=security_shift_df,
        output_dir=output_dir,
    )

    save_shap_summary_plots(
        rankings=rankings,
        output_dir=os.path.join(output_dir, "plots"),
        max_display=15,
    )

    return {
        "X_sample": X_sample,
        "y_sample": y_sample,
        "shap_values_dict": shap_values_dict,
        "rankings": rankings,
        "global_spearman_df": global_spearman_df,
        "class_specific_rankings": class_specific_rankings,
        "class_spearman_df": class_spearman_df,
        "spearman_summary_df": spearman_summary_df,
        "security_shift_df": security_shift_df,
    }