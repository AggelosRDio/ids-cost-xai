# ==================================================
# SETUP
# ==================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

from imblearn.over_sampling import SMOTE


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

print("Classes:", le.classes_)

n_classes = len(le.classes_)


# ==================================================
# SCALING (IMPORTANT FOR ISOLATION FOREST)
# ==================================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)


# ==================================================
# SMOTE
# ==================================================
smote = SMOTE(
    sampling_strategy={
        2: 15000,   # Probe
        3: 5000,    # R2L
        4: 2000     # U2R
    },
    random_state=42
)

X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_enc)


# ==================================================
# MAIN MODEL
# ==================================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)


# ==================================================
# CALIBRATION
# ==================================================
cal_model = CalibratedClassifierCV(
    model,
    method="isotonic",
    cv=5
)

cal_model.fit(X_val_scaled, y_val_enc)


# ==================================================
# ANOMALY DETECTOR (FIXED ALIGNMENT)
# ==================================================
normal_class = "Normal" if "Normal" in y_train.values else y_train.value_counts().idxmax()

normal_mask = (y_train.values == normal_class)

iso = IsolationForest(
    contamination=0.02,
    random_state=42
)

iso.fit(X_train_scaled[normal_mask])


# ==================================================
# COST MATRIX (STABILIZED VERSION)
# ==================================================
cost_matrix = np.array([
    [0, 1, 2, 5, 10],
    [1, 0, 2, 5, 10],
    [2, 2, 0, 4, 8],
    [6, 6, 4, 0, 4],
    [10,10,8,4,  0]
])

assert cost_matrix.shape == (n_classes, n_classes), "Cost matrix mismatch!"


# ==================================================
# HYBRID PREDICTION (FIXED + STABLE)
# ==================================================
def hybrid_predict(X):

    probs = cal_model.predict_proba(X)
    preds = []

    for i in range(len(X)):

        sample = X[i:i+1]

        # =====================
        # ANOMALY DETECTION
        # =====================
        anomaly_flag = iso.predict(sample)[0]

        sample_probs = probs[i].copy()

        # Soft anomaly influence (NO HARD OVERRIDE)
        if anomaly_flag == -1:
            sample_probs = sample_probs + np.array([0.0, 0.0, 0.0, 0.80, 0.90])

        # normalize safely
        sample_probs = sample_probs / (np.sum(sample_probs) + 1e-12)

        # =====================
        # COST-SENSITIVE DECISION
        # =====================
        expected_costs = cost_matrix.T @ sample_probs

        best_class = int(np.argmin(expected_costs))
        preds.append(le.inverse_transform([best_class])[0])

    return np.array(preds)


# ==================================================
# VALIDATION
# ==================================================
print("\n================ VALIDATION ================\n")

val_preds = hybrid_predict(X_val_scaled)
print(classification_report(y_val, val_preds))


# ==================================================
# TEST
# ==================================================
print("\n================ TEST ================\n")

test_preds = hybrid_predict(X_test_scaled)
print(classification_report(y_test, test_preds))


# ==================================================
# CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(
    y_test,
    test_preds,
    labels=le.classes_
)

print("\nConfusion Matrix:\n")
print(cm)


# ==================================================
# CRITICAL CLASSES
# ==================================================
report = classification_report(y_test, test_preds, output_dict=True)

print("\n================ CRITICAL RESULTS ================\n")
print("R2L Recall:", report.get("R2L", {}).get("recall", 0))
print("U2R Recall:", report.get("U2R", {}).get("recall", 0))