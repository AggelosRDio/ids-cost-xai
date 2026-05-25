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
from sklearn.model_selection import train_test_split 
from sklearn.frozen import FrozenEstimator
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
# BASE MODEL
# ==================================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


model.fit(X_train, y_train_enc)

# ==================================================
# SPLIT VALIDATION SET 
# ==================================================

X_calib, X_val_eval, y_calib_enc, y_val_eval_enc = train_test_split(
    X_val, y_val_enc, test_size=0.5, random_state=42, stratify=y_val_enc
)

# ==================================================
# PROBABILITY CALIBRATION 
# ==================================================
frozen_model = FrozenEstimator(model)
cal_model = CalibratedClassifierCV(
    frozen_model,
    method="sigmoid",
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
# VALIDATION EVALUATION
# ==================================================
print("\n================ VALIDATION ================\n")


y_val_pred_enc = predict_cost_sensitive(cal_model, X_val_eval, cost_matrix)


y_val_pred = le.inverse_transform(y_val_pred_enc)
y_val_eval = le.inverse_transform(y_val_eval_enc)

print(classification_report(y_val_eval, y_val_pred))

# ==================================================
# TEST EVALUATION
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