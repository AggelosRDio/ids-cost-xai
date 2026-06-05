"""
Real-world grounded cost matrix for IDS misclassification.
 
Cost derivation sources:
  - IBM Cost of a Data Breach Report 2024 (average breach costs by type)
  - Ponemon Institute SOC Analyst Cost Study (FP triage time)
  - NIST SP 800-30 Rev.1 (Risk = Likelihood × Impact framework)
  - CVSS v3.1 Base Scores (severity proxy per attack category)
 
Economic assumptions (documented for paper):
  U2R  (Unauthorized Root):   $3.15M  = $4.5M breach cost × 0.70 escalation prob.
  R2L  (Remote-to-Local):     $1.20M  = $2.0M breach cost × 0.60 escalation prob.
  DoS  (Denial of Service):   $560K   = $1.4M breach cost × 0.40 escalation prob.
  Probe (Reconnaissance):     $85K    = $0.21M breach cost × 0.40 escalation prob.
  FP   (analyst triage):      $12.50  = 15 min × $50/hr SOC analyst rate
 
Raw ratios are log-compressed for gradient stability during training.
Three regimes scale the moderate baseline up/down for ablation study.
"""

import numpy as np

RAW_FN_COSTS = {
    "U2R":   3_150_000,
    "R2L":   1_200_000,
    "DoS":     560_000,
    "Probe":    85_000,
}

RAW_FP_COST = 12.50

REGIME_SCALES = {
    "conservative": 0.20,
    "moderate":     1.00,
    "aggressive":   2.00
}
DEFAULT_REGIME = "moderate"

# DoS=0, Normal=1, Probe=2, R2L=3, U2R=4
CLASS_TO_IDX = {"DoS": 0, "Normal": 1, "Probe": 2, "R2L": 3, "U2R": 4}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

def _log_compress(ratio: float) -> float:
    return float(np.log1p(ratio))

def get_sample_weights(y_train_res: np.ndarray, regime: str = DEFAULT_REGIME) -> np.ndarray:
    """
    Derive per sample training weights from real-world costs. Weights are log-compressed rations of FN cost / FP cost scaled by the chosen regime factor.

    This ensures sample_weight and cost_matrix are always derived from the same economic sources and they cannot drift apart.
    """

    scale = REGIME_SCALES[regime]
    weights = np.ones(len(y_train_res), dtype=float)

    for cls, fn_cost in RAW_FN_COSTS.items():
        raw_ratio = fn_cost / RAW_FP_COST
        compressed = _log_compress(raw_ratio) * scale
        idx = CLASS_TO_IDX[cls]
        weights[y_train_res == idx] = compressed

    return weights

def get_cost_matrix(regime: str = DEFAULT_REGIME) -> np.ndarray:
    """
    Builds a 5×5 cost matrix for cost-sensitive inference.
 
    cost_matrix[predicted, true] = cost of predicting `predicted`
    when the true class is `true`.
 
    Diagonal = 0 (correct prediction).
    Off-diagonal = proportional to the FN cost of the true class,
    normalized so FP (misclassifying attack as Normal) = 1.0 baseline,
    and FN costs scale from that anchor.
 
    Three regimes (conservative/moderate/aggressive) control how
    aggressively the asymmetry is applied.
    """
    scale = REGIME_SCALES[regime]
    n = len(CLASS_TO_IDX)
    matrix = np.zeros((n, n), dtype=float)

    # Anchor: FP cost = 1 unit
    # FN costs are expressed as multiples of the FP cost, log-compressed
    for true_idx in range(n):
        true_cls = IDX_TO_CLASS[true_idx]
        if true_cls == "Normal":
            fn_ratio = 1.0   # misclassifying Normal as attack → FP, low cost
        else:
            fn_ratio = _log_compress(RAW_FN_COSTS[true_cls] / RAW_FP_COST) * scale
 
        for pred_idx in range(n):
            if pred_idx == true_idx:
                matrix[pred_idx, true_idx] = 0.0
            else:
                matrix[pred_idx, true_idx] = fn_ratio
 
    return matrix


def predict_cost_sensitive(model, X: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """
    Vectorized cost-sensitive prediction.
    Selects the class that minimises expected cost under the model's
    probability estimates.
 
    Args:
        model:       fitted sklearn-compatible classifier with predict_proba
        X:           feature matrix (n_samples, n_features)
        cost_matrix: (n_classes, n_classes) cost[pred, true]
 
    Returns:
        predicted class indices (n_samples,)
    """
    probs = model.predict_proba(X)
    expected_costs = probs @ cost_matrix.T
    return np.argmin(expected_costs, axis=1)


def total_economic_loss(y_true_enc: np.ndarray, y_pred_enc: np.ndarray) -> dict:
    """
    Compute Total Economic Loss using raw USD costs (not the compressed matrix).
 
    TEL = Σ Cost(FN, class) for each missed attack
        + Σ Cost(FP) for each false alarm
 
    Returns a dict with total TEL and per-error-type breakdown.
    """
    tel_fn, tel_fp = 0.0, 0.0
    fn_breakdown = {cls: 0.0 for cls in RAW_FN_COSTS}
 
    for true_enc, pred_enc in zip(y_true_enc, y_pred_enc):
        if true_enc == pred_enc:
            continue
        true_cls = IDX_TO_CLASS[true_enc]
        pred_cls = IDX_TO_CLASS[pred_enc]
 
        if true_cls != "Normal" and pred_cls == "Normal":
            # False Negative: missed attack
            cost = RAW_FN_COSTS[true_cls]
            tel_fn += cost
            fn_breakdown[true_cls] += cost
        elif true_cls == "Normal":
            # False Positive: false alarm
            tel_fp += RAW_FP_COST
 
    return {
        "TEL":          tel_fn + tel_fp,
        "TEL_FN":       tel_fn,
        "TEL_FP":       tel_fp,
        "FN_breakdown": fn_breakdown,
    }