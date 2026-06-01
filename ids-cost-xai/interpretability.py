
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ==================================================
# LOAD DATA
# ==================================================

train = pd.read_csv("data/nslkdd/processed/nslkdd_train.csv")
val   = pd.read_csv("data/nslkdd/processed/nslkdd_validation.csv")
test  = pd.read_csv("data/nslkdd/processed/nslkdd_test.csv")

target = "macro_label"

X_train = train.drop(columns=[target])
y_train = train[target]

X_val = val.drop(columns=[target])
y_val = val[target]

X_test = test.drop(columns=[target])
y_test = test[target]

# ==================================================
# LABEL ENCODING
# ==================================================

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)


##αφαιρεση χαραξτηριστικων που δεν βοηθουν στην διακριση των κλασεων και εχουν πολυ χαμηλη συχνοτητα 
cols_to_drop = ['num_outbound_cmds', 'is_host_login', 'is_guest_login']
X_train = X_train.drop(columns=cols_to_drop)
X_val = X_val.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)

###Log transformation για να μειωσει την skewness των χαρακτηριστικων που εχουν μεγαλη διασπορα και skewness, βοηθαει το μοντελο να μαθει καλυτερα τις κλασεις που εχουν λιγοτερα δειγματα.
skewed_features = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 
                   'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations']

# Εφαρμόζουμε log(1+x) για να αποφύγουμε το log(0)
for col in skewed_features:
    if col in X_train.columns:
        X_train[col] = np.log1p(X_train[col])
        X_val[col]   = np.log1p(X_val[col])
        X_test[col]  = np.log1p(X_test[col])

# ==================================================
# SMOTE (CONTROLLED)
# ==================================================

scaler=MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


X_calib, X_val_eval, y_calib_enc, y_val_eval_enc = train_test_split(
    X_val, y_val_enc, test_size=0.5, random_state=42, stratify=y_val_enc
)


smote = SMOTE(
    sampling_strategy={
    2:15000,
    3:20000,
    4:5000
},
    random_state=42
)

smote_enn = SMOTEENN(
    smote=smote,
    random_state=42
)

X_train_res, y_train_res = smote_enn.fit_resample(
    X_train,
    y_train_enc
)
# ==================================================
# BASE MODEL
# ==================================================

inv_freq_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_res),
    y=y_train_res
)

fn_costs = cost_matrix[1, :].astype(float) 
fn_costs[1] = 1.0 
hybrid_fn_weights = inv_freq_weights * fn_costs
class_weights = {i: float(w / hybrid_fn_weights.min()) for i, w in enumerate(hybrid_fn_weights)}


model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res, sample_weight=np.array([class_weights[i] for i in y_train_res]))

# ==================================================
# CALIBRATION (IMPORTANT)
# ==================================================
frozen_model = FrozenEstimator(model)
cal_model = CalibratedClassifierCV(
    frozen_model,
    method="sigmoid"
)

cal_model.fit(X_calib, y_calib_enc)

# ==================================================
# COST MATRIX
# ==================================================
cost_matrix = np.array([
    [0, 1, 2, 5, 10],
    [1, 0, 2, 5, 10],
    [2, 2, 0, 5, 10],
    [5, 5, 3, 0, 10],
    [10,10,10,5,  0]
])
# ==================================================
# COST-SENSITIVE PREDICTION
# ==================================================
def predict_cost_sensitive(model, X, cost_matrix):

    probs = model.predict_proba(X)
    preds = []

    for i in range(len(X)):

        sample_probs = probs[i]
        costs = []

        for c in range(len(cost_matrix)):
            expected_cost = np.sum(sample_probs * cost_matrix[c])
            costs.append(expected_cost)

        best_class = np.argmin(costs)
        preds.append(best_class)

    return np.array(preds)

# ==================================================
# VALIDATION
# ==================================================
y_val_pred_enc = predict_cost_sensitive(cal_model, X_val_eval, cost_matrix)
y_val_pred = le.inverse_transform(y_val_pred_enc)
y_val_eval = le.inverse_transform(y_val_eval_enc)
# ==================================================
# TEST
# ==================================================
y_test_pred_enc = predict_cost_sensitive(cal_model, X_test, cost_matrix)
y_test_pred = le.inverse_transform(y_test_pred_enc)


# ==================================================
# INTERPRETABILITY SECTION
# ==================================================



X_test_human = test.drop(
    columns=[
        target,
        'num_outbound_cmds',
        'is_host_login',
        'is_guest_login'
    ]
).copy()

feature_names = X_test_human.columns.tolist()

# ==================================================
# PREDICTIONS
# ==================================================

y_probs = cal_model.predict_proba(X_test)


y_simple_preds = np.argmax(y_probs, axis=1)

# ==================================================
# SHAP EXPLAINER
# ==================================================


explainer = shap.TreeExplainer(model)

# ==================================================
# FUNCTION FOR LOCAL INTERPRETABILITY
# ==================================================

def explain_corrected_sample(target_class_idx, class_name_label):

    corrected_samples = np.where(
        (y_simple_preds != y_test_enc) &
        (y_test_pred_enc == y_test_enc) &
        (y_test_enc == target_class_idx)
    )[0]

    if len(corrected_samples) == 0:

        print(
            f"\nNo corrected samples found for "
            f"{class_name_label}."
        )

        return

    target_idx = corrected_samples[0]

    actual_class_idx = y_test_enc[target_idx]

    actual_class_name = le.inverse_transform(
        [actual_class_idx]
    )[0]

    simple_guess_name = le.inverse_transform(
        [y_simple_preds[target_idx]]
    )[0]

    final_decision_name = le.inverse_transform(
        [y_test_pred_enc[target_idx]]
    )[0]

    print("\n" + "="*70)
    print(f"{class_name_label} COST-SENSITIVE CORRECTION")
    print("="*70)

    print(f"Sample index: {target_idx}")
    print(f"True class: {actual_class_name}")
    print(f"Raw XGBoost prediction: {simple_guess_name}")
    print(f"Final cost-sensitive decision: {final_decision_name}")

    # ==================================================
    # CLASS PROBABILITIES
    # ==================================================

    print("\nClass probabilities:")

    for i, prob in enumerate(y_probs[target_idx]):

        cname = le.inverse_transform([i])[0]

        print(f"{cname:<12} -> {prob:.4f}")

    # ==================================================
    # EXPECTED COSTS
    # ==================================================

    print("\nExpected cost per decision:")

    sample_probs = y_probs[target_idx]

    for c in range(len(cost_matrix)):

        expected_cost = np.sum(
            sample_probs * cost_matrix[c]
        )

        cname = le.inverse_transform([c])[0]

        print(f"{cname:<12} -> {expected_cost:.4f}")

    # ==================================================
    # SHAP VALUES
    # ==================================================

    print("\nComputing SHAP explanations...")

    shap_values_single = explainer.shap_values(
        X_test[[target_idx]]
    )

    if isinstance(shap_values_single, list):

        shap_for_class = shap_values_single[
            actual_class_idx
        ][0]

    else:

        shap_for_class = shap_values_single[
            0, :, actual_class_idx
        ]

    # ==================================================
    # WATERFALL PLOT
    # ==================================================

    exp = shap.Explanation(
        values=shap_for_class,
        base_values=explainer.expected_value[
            actual_class_idx
        ],
        data=X_test_human.iloc[target_idx].values,
        feature_names=feature_names
    )

    plt.figure(figsize=(13, 8))

    shap.waterfall_plot(
        exp,
        max_display=10,
        show=False
    )

    plt.title(
        f"{class_name_label} Cost-Sensitive Decision Correction\n"
        f"Sample {target_idx}: "
        f"{simple_guess_name} → {actual_class_name}",
        fontsize=13
    )

    plt.tight_layout()
    plt.show()

# ==================================================
# FUNCTION FOR GLOBAL INTERPRETABILITY
# ==================================================

def global_explainability(target_class_idx, class_name_label):

    print(
        f"\nComputing global SHAP explanations "
        f"for {class_name_label}..."
    )

   
    X_sample_scaled = X_test[:300]

   
    X_sample_human = X_test_human.iloc[:300]

    shap_values_global = explainer.shap_values(
        X_sample_scaled
    )

    if isinstance(shap_values_global, list):

        vals_to_plot = shap_values_global[
            target_class_idx
        ]

    else:

        vals_to_plot = shap_values_global[
            :, :, target_class_idx
        ]

    # ==================================================
    # SUMMARY PLOT
    # ==================================================

    plt.figure(figsize=(11, 7))

    shap.summary_plot(
        vals_to_plot,
        X_sample_human,
        show=False
    )

    plt.title(
        f"SHAP Summary Plot for "
        f"{class_name_label} Attack Detection",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

    # ==================================================
    # BAR PLOT
    # ==================================================

    plt.figure(figsize=(11, 7))

    shap.summary_plot(
        vals_to_plot,
        X_sample_human,
        plot_type="bar",
        show=False
    )

    plt.title(
        f"Global Feature Importance for "
        f"{class_name_label} Classification",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

# ==================================================
# R2L INTERPRETABILITY
# ==================================================

R2L_CLASS_INDEX = 3

explain_corrected_sample(
    R2L_CLASS_INDEX,
    "R2L"
)

global_explainability(
    R2L_CLASS_INDEX,
    "R2L"
)
