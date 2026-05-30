# ==================================================
# SETUP
# ==================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split

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
# PREPROCESSING
# ==================================================

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
# mapping:
# DoS=0, Normal=1, Probe=2, R2L=3, U2R=4

scaler=MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


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



print("\nAfter SMOTE:")
unique, counts = np.unique(y_train_res, return_counts=True)
for u, c in zip(unique, counts):
    print(le.inverse_transform([u])[0], ":", c)

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
# BASE MODEL
# ==================================================

row_sums = cost_matrix.sum(axis=1)
class_weights = {i: weight for i, weight in enumerate(row_sums / row_sums.min())}

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=5,
    max_features="sqrt",
    bootstrap=True,
    class_weight=class_weights,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)

# ==================================================
# SPLIT VALIDATION SET 
# ==================================================
X_calib, X_val_eval, y_calib_enc, y_val_eval_enc = train_test_split(
    X_val, y_val_enc, test_size=0.5, random_state=42, stratify=y_val_enc
)

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

y_val_pred_enc = predict_cost_sensitive(cal_model, X_val_eval, cost_matrix)
y_val_pred = le.inverse_transform(y_val_pred_enc)
y_val_eval = le.inverse_transform(y_val_eval_enc)
print(classification_report(y_val_eval, y_val_pred))


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
