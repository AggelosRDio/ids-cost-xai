# ==================================================
# CLEAN ENV
# ==================================================
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")


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
# LABEL ENCODER (stage 2)
# ==================================================
le = LabelEncoder()
le.fit(y_train)


# ==================================================
# STAGE 1: Normal vs Attack
# ==================================================
y_train_s1 = (y_train != "Normal").astype(int)

stage1 = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=1
)

stage1.fit(X_train, y_train_s1)


# ==================================================
# STAGE 2: Multi-class Attack Classifier
# ==================================================
mask = (y_train != "Normal")

X_train_s2 = X_train[mask]
y_train_s2 = y_train[mask]

y_train_s2_enc = le.transform(y_train_s2)

stage2 = RandomForestClassifier(
    n_estimators=400,
    class_weight={
        le.transform(["DoS"])[0]: 1,
        le.transform(["Probe"])[0]: 2,
        le.transform(["R2L"])[0]: 6,
        le.transform(["U2R"])[0]: 12
    },
    random_state=42,
    n_jobs=1
)

stage2.fit(X_train_s2, y_train_s2_enc)


# ==================================================
# STAGE 3: SPECIALIZED BINARY DETECTORS
# ==================================================
# R2L detector
y_train_r2l = (y_train == "R2L").astype(int)

r2l_clf = RandomForestClassifier(
    n_estimators=300,
    class_weight={0:1, 1:20},
    random_state=42,
    n_jobs=1
)

r2l_clf.fit(X_train, y_train_r2l)

# U2R detector
y_train_u2r = (y_train == "U2R").astype(int)

u2r_clf = RandomForestClassifier(
    n_estimators=300,
    class_weight={0:1, 1:50},
    random_state=42,
    n_jobs=1
)

u2r_clf.fit(X_train, y_train_u2r)


# ==================================================
# FINAL PREDICTION FUNCTION
# ==================================================
def predict(X, t_stage1=0.4, t_r2l=0.2, t_u2r=0.1):

    preds = []

    for i in range(len(X)):

        x = X.iloc[i:i+1]

        # ------------------------
        # Stage 1
        # ------------------------
        attack_prob = stage1.predict_proba(x)[0][1]

        if attack_prob < t_stage1:
            preds.append("Normal")
            continue

        # ------------------------
        # Stage 2
        # ------------------------
        cls = stage2.predict(x)[0]
        label = le.inverse_transform([cls])[0]

        # ------------------------
        # Stage 3 (binary refinement)
        # ------------------------
        r2l_prob = r2l_clf.predict_proba(x)[0][1]
        u2r_prob = u2r_clf.predict_proba(x)[0][1]

        # override ONLY if confident
        if r2l_prob > t_r2l:
            preds.append("R2L")
        elif u2r_prob > t_u2r:
            preds.append("U2R")
        else:
            preds.append(label)

    return np.array(preds)


# ==================================================
# VALIDATION TUNING
# ==================================================
print("\n================ VALIDATION TUNING ================\n")

best_score = -1

for t1 in [0.3, 0.4, 0.5]:
    for t_r2l in [0.1, 0.2, 0.3]:
        for t_u2r in [0.05, 0.1, 0.2]:

            pred = predict(X_val, t1, t_r2l, t_u2r)

            report = classification_report(y_val, pred, output_dict=True)

            score = (
                report.get("R2L", {}).get("recall", 0) +
                report.get("U2R", {}).get("recall", 0)
            )

            if score > best_score:
                best_score = score
                best_params = (t1, t_r2l, t_u2r)

print("BEST PARAMS:", best_params)


# ==================================================
# FINAL VALIDATION
# ==================================================
print("\n================ FINAL VALIDATION ================\n")

val_pred = predict(X_val, *best_params)
print(classification_report(y_val, val_pred))


# ==================================================
# FINAL TEST
# ==================================================
print("\n================ FINAL TEST ================\n")

test_pred = predict(X_test, *best_params)
print(classification_report(y_test, test_pred))


# ==================================================
# CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(
    y_test,
    test_pred,
    labels=["Normal","DoS","Probe","R2L","U2R"]
)

print("\nConfusion Matrix:\n")
print(cm)


# ==================================================
# CRITICAL METRICS
# ==================================================
report = classification_report(y_test, test_pred, output_dict=True)

print("\n================ CRITICAL RESULTS ================\n")
print("R2L Recall:", report.get("R2L", {}).get("recall", 0))
print("U2R Recall:", report.get("U2R", {}).get("recall", 0))