import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from scipy.stats import spearmanr
from xgboost import XGBClassifier


DEFAULT_SECURITY_FEATURES = [
    "root_shell",
    "su_attempted",
    "num_failed_logins",
    "num_shells",
    "num_access_files",
    "logged_in",
    "num_compromised",
    "num_root",
    "num_file_creations",
]


def ensure_dataframe(X, feature_names):
    """
    Ensure that the feature matrix is a pandas DataFrame with readable feature names.
    This is useful because SHAP plots and rankings are more interpretable with named columns.
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()

    return pd.DataFrame(X, columns=feature_names)


def make_shap_sample(X, feature_names, sample_size=1000, random_state=42):
    """
    Create a fixed test sample for SHAP analysis.

    SHAP can be computationally expensive, especially for multiclass models.
    A fixed random sample makes the analysis faster and reproducible.
    """
    X_df = ensure_dataframe(X, feature_names)

    if sample_size is None or sample_size >= len(X_df):
        return X_df

    return X_df.sample(n=sample_size, random_state=random_state)


def train_raw_cost_sensitive_xgb_models(
    regimes,
    X_train,
    y_train,
    get_sample_weights_func,
    random_state=42,
    xgb_params=None,
):
    """
    Train raw cost-sensitive XGBoost models for SHAP analysis.

    Calibrated models are useful for probability calibration and evaluation,
    but SHAP TreeExplainer works more directly and reliably with the raw tree model.
    Therefore, we retrain raw XGBClassifier models for each cost regime.
    """
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "objective": "multi:softprob",
            "num_class": 5,
            "eval_metric": "mlogloss",
            "random_state": random_state,
            "n_jobs": -1,
        }

    raw_models = {}

    for regime in regimes:
        print(f"[XAI] Training raw XGB model for SHAP — regime: {regime}")

        sample_weights = get_sample_weights_func(y_train, regime=regime)

        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        raw_models[regime] = model

    return raw_models


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


def compute_all_shap_rankings(baseline_model, raw_cost_models, X_sample, feature_names):
    """
    Compute SHAP values and global rankings for the baseline and cost-sensitive models.
    """
    shap_values_dict = {}
    rankings = {}

    shap_values_dict["baseline"], rankings["baseline"] = compute_shap_ranking(
        baseline_model,
        X_sample,
        feature_names,
        "baseline",
    )

    for regime, model in raw_cost_models.items():
        shap_values_dict[regime], rankings[regime] = compute_shap_ranking(
            model,
            X_sample,
            feature_names,
            regime,
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
    baseline_model,
    regimes,
    X_test,
    feature_names,
    X_train_res,
    y_train_res,
    get_sample_weights_func,
    encoder,
    random_state=42,
    sample_size=1000,
    output_dir="results/xai",
    target_classes=("R2L", "U2R"),
    security_features=None,
):
    """
    End-to-end XAI audit for the cost-sensitive IDS pipeline.

    This function:
    1. Creates a fixed SHAP sample.
    2. Trains raw cost-sensitive XGBoost models for SHAP.
    3. Computes global SHAP rankings.
    4. Computes global Spearman rank correlations.
    5. Tracks security-critical feature shifts.
    6. Computes class-specific SHAP rankings for R2L and U2R.
    7. Computes class-specific Spearman correlations.
    8. Saves result tables and plots.
    """
    X_sample = make_shap_sample(
        X_test,
        feature_names,
        sample_size=sample_size,
        random_state=random_state,
    )

    raw_cost_models = train_raw_cost_sensitive_xgb_models(
        regimes=regimes,
        X_train=X_train_res,
        y_train=y_train_res,
        get_sample_weights_func=get_sample_weights_func,
        random_state=random_state,
    )

    shap_values_dict, rankings = compute_all_shap_rankings(
        baseline_model=baseline_model,
        raw_cost_models=raw_cost_models,
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
        "raw_cost_models": raw_cost_models,
        "shap_values_dict": shap_values_dict,
        "rankings": rankings,
        "global_spearman_df": global_spearman_df,
        "class_specific_rankings": class_specific_rankings,
        "class_spearman_df": class_spearman_df,
        "spearman_summary_df": spearman_summary_df,
        "security_shift_df": security_shift_df,
    }
