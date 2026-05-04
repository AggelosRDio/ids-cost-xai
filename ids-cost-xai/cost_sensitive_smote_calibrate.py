# ==================================================
# SETUP
# ==================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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


# ==================================================
# CHECK CLASS COUNTS
# ==================================================
unique, counts = np.unique(y_train_enc, return_counts=True)
print("\nOriginal distribution:")
for u, c in zip(unique, counts):
    print(le.inverse_transform([u])[0], ":", c)


# ==================================================
# SMOTE (CONTROLLED)
# ==================================================
# mapping:
# DoS=0, Normal=1, Probe=2, R2L=3, U2R=4

smote = SMOTE(
    sampling_strategy={
        2: 15000,   # Probe
        3: 5000,    # R2L
        4: 2000     # U2R
    },
    random_state=42
)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train_enc)

print("\nAfter SMOTE:")
unique, counts = np.unique(y_train_res, return_counts=True)
for u, c in zip(unique, counts):
    print(le.inverse_transform([u])[0], ":", c)


# ==================================================
# BASE MODEL
# ==================================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)


# ==================================================
# CALIBRATION (IMPORTANT)
# ==================================================
cal_model = CalibratedClassifierCV(
    model,
    method="isotonic",
    cv=5
)

cal_model.fit(X_val, y_val_enc)


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
print("\n================ VALIDATION ================\n")

y_val_pred_enc = predict_cost_sensitive(cal_model, X_val, cost_matrix)
y_val_pred = le.inverse_transform(y_val_pred_enc)

print(classification_report(y_val, y_val_pred))


# ==================================================
# TEST
# ==================================================
print("\n================ TEST ================\n")

y_test_pred_enc = predict_cost_sensitive(cal_model, X_test, cost_matrix)
y_test_pred = le.inverse_transform(y_test_pred_enc)

print(classification_report(y_test, y_test_pred))


# ==================================================
# CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=le.classes_
)

print("\nConfusion Matrix:\n")
print(cm)


# ==================================================
# CRITICAL CLASSES
# ==================================================
report = classification_report(y_test, y_test_pred, output_dict=True)

print("\n================ CRITICAL RESULTS ================\n")
print("R2L Recall:", report.get("R2L", {}).get("recall", 0))
print("U2R Recall:", report.get("U2R", {}).get("recall", 0))