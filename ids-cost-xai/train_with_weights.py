# ==================================================
# CLEAN ENV (NO WARNINGS)
# ==================================================
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from joblib import parallel_backend
from imblearn.over_sampling import SMOTE

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["PYTHONWARNINGS"] = "ignore"

warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)


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
# LABEL ENCODING (multi-class stage 2)
# ==================================================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)


# ==================================================
# STAGE 1: BINARY (Normal vs Attack)
# ==================================================
y_train_stage1 = (y_train != "Normal").astype(int)
y_val_stage1   = (y_val != "Normal").astype(int)
y_test_stage1  = (y_test != "Normal").astype(int)

stage1 = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=1
)

stage1.fit(X_train, y_train_stage1)


# ==================================================
# STAGE 2 TRAIN DATA (ONLY ATTACKS)
# ==================================================
mask = (y_train != "Normal")

X_train_s2 = X_train.loc[mask]
y_train_s2 = y_train.loc[mask]

y_train_s2_enc = le.fit_transform(y_train_s2)


# ==================================================
# SMOTE (SAFE PARALLEL)
# ==================================================
smote = SMOTE(random_state=42)

with parallel_backend("threading"):
    X_res, y_res = smote.fit_resample(X_train_s2, y_train_s2_enc)


# ==================================================
# COST-SENSITIVE MATRIX
# order: [Normal, DoS, Probe, R2L, U2R]
# ==================================================
cost_matrix = np.array([
    [0, 1, 2, 5, 10],
    [1, 0, 2, 5, 10],
    [2, 2, 0, 5, 10],
    [5, 5, 3, 0, 10],
    [10,10,10,5,  0]
])


# ==================================================
# STAGE 2 MODEL (COST-AWARE)
# ==================================================
stage2 = RandomForestClassifier(
    n_estimators=400,
    class_weight={
        0: 1,
        1: 1,
        2: 2,
        3: 5,
        4: 10
    },
    random_state=42,
    n_jobs=1
)

stage2.fit(X_res, y_res)


# ==================================================
# COST-SENSITIVE PREDICTION FUNCTION
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
# THRESHOLD TUNING (optional improvement)
# ==================================================
print("\n================ VALIDATION TUNING ================\n")

best_score = -1
best_model = None

for t in [0.4, 0.5, 0.6]:

    # stage 1 filter
    probs = stage1.predict_proba(X_val)[:, 1]
    attack_mask = probs > t

    preds = []

    for i, is_attack in enumerate(attack_mask):

        if not is_attack:
            preds.append("Normal")
        else:
            row = X_val.iloc[i:i+1]
            cls = stage2.predict(row)[0]
            preds.append(le.inverse_transform([cls])[0])

    report = classification_report(y_val, preds, output_dict=True)

    score = report.get("R2L", {}).get("recall", 0) + report.get("U2R", {}).get("recall", 0)

    print(f"Threshold {t} | R2L={report.get('R2L', {}).get('recall',0):.3f} | U2R={report.get('U2R', {}).get('recall',0):.3f}")

    if score > best_score:
        best_score = score
        best_t = t


print("\nBEST THRESHOLD:", best_t)


# ==================================================
# FINAL VALIDATION
# ==================================================
print("\n================ FINAL VALIDATION ================\n")

probs = stage1.predict_proba(X_val)[:, 1]
attack_mask = probs > best_t

final_preds = []

for i, is_attack in enumerate(attack_mask):

    if not is_attack:
        final_preds.append("Normal")
    else:
        row = X_val.iloc[i:i+1]
        cls = stage2.predict(row)[0]
        final_preds.append(le.inverse_transform([cls])[0])

print(classification_report(y_val, final_preds))


# ==================================================
# TEST EVALUATION
# ==================================================
print("\n================ FINAL TEST ================\n")

probs = stage1.predict_proba(X_test)[:, 1]
attack_mask = probs > best_t

final_preds = []

for i, is_attack in enumerate(attack_mask):

    if not is_attack:
        final_preds.append("Normal")
    else:
        row = X_test.iloc[i:i+1]
        cls = stage2.predict(row)[0]
        final_preds.append(le.inverse_transform([cls])[0])

print(classification_report(y_test, final_preds))


# ==================================================
# CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(y_test, final_preds,
                      labels=["Normal","DoS","Probe","R2L","U2R"])

print("\nConfusion Matrix:\n")
print(cm)


# ==================================================
# CRITICAL CLASSES
# ==================================================
report = classification_report(y_test, final_preds, output_dict=True)

print("\n================ CRITICAL RESULTS ================\n")
print("R2L Recall:", report.get("R2L", {}).get("recall", 0))
print("U2R Recall:", report.get("U2R", {}).get("recall", 0))