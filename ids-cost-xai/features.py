# ==================================================
# SETUP
# ==================================================
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier


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

classes = le.classes_
print("Classes:", classes)

normal_class = le.transform(["Normal"])[0]


# ==================================================
# SCALE
# ==================================================
scaler = MinMaxScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val   = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# ==================================================
# STAGE 1: NORMAL vs ATTACK
# ==================================================
y_train_stage1 = (y_train_enc != normal_class).astype(int)
y_val_stage1   = (y_val_enc != normal_class).astype(int)
y_test_stage1  = (y_test_enc != normal_class).astype(int)


stage1 = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

stage1.fit(X_train, y_train_stage1)


# ==================================================
# STAGE 2: ATTACK CLASSIFIER (FIXED LABELS)
# ==================================================
attack_mask = y_train_enc != normal_class

X_train_att = X_train[attack_mask]
y_train_att = y_train_enc[attack_mask]

# 🔥 FIX LABEL SPACE
unique_classes = np.unique(y_train_att)
mapping = {old:i for i, old in enumerate(unique_classes)}
reverse_mapping = {v:k for k,v in mapping.items()}

y_train_att_fixed = np.array([mapping[y] for y in y_train_att])


stage2 = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(unique_classes),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

stage2.fit(X_train_att, y_train_att_fixed)


# ==================================================
# COST MATRIX
# ==================================================
cost_matrix = np.array([
    [0, 1, 2, 5, 10],
    [1, 0, 2, 5, 10],
    [2, 2, 0, 5, 10],
    [5, 5, 3, 0, 10],
    [10,10,10,5, 0]
])


# ==================================================
# STAGE 1 PREDICTION (cost-aware threshold)
# ==================================================
def stage1_predict(X, threshold=0.3):

    probs = stage1.predict_proba(X)[:, 1]
    return (probs > threshold).astype(int)


# ==================================================
# STAGE 2 COST-SENSITIVE PREDICTION
# ==================================================
def stage2_predict_cost_sensitive(X):

    probs = stage2.predict_proba(X)
    preds = []

    for i in range(len(X)):

        p = probs[i]
        costs = []

        for c in range(len(cost_matrix)):
            costs.append(np.sum(p * cost_matrix[c]))

        preds.append(np.argmin(costs))

    # map back to original labels
    preds = [reverse_mapping[p] for p in preds]

    return np.array(preds)


# ==================================================
# FINAL PREDICTION PIPELINE
# ==================================================
def predict(X):

    is_attack = stage1_predict(X, threshold=0.3)

    final_preds = []

    for i in range(len(X)):

        if is_attack[i] == 0:
            final_preds.append(normal_class)

        else:
            pred = stage2_predict_cost_sensitive(X.iloc[[i]])[0]
            final_preds.append(pred)

    return np.array(final_preds)


# ==================================================
# VALIDATION
# ==================================================
print("\n================ VALIDATION ================\n")

y_val_pred = predict(X_val)

print(classification_report(y_val, le.inverse_transform(y_val_pred)))


# ==================================================
# TEST
# ==================================================
print("\n================ TEST ================\n")

y_test_pred = predict(X_test)

print(classification_report(y_test, le.inverse_transform(y_test_pred)))


# ==================================================
# CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(
    y_test,
    le.inverse_transform(y_test_pred),
    labels=le.classes_
)

print("\nConfusion Matrix:\n")
print(cm)


# ==================================================
# CRITICAL METRICS
# ==================================================
report = classification_report(
    y_test,
    le.inverse_transform(y_test_pred),
    output_dict=True
)

print("\n================ CRITICAL RESULTS ================\n")
print("R2L Recall:", report.get("R2L", {}).get("recall", 0))
print("U2R Recall:", report.get("U2R", {}).get("recall", 0))