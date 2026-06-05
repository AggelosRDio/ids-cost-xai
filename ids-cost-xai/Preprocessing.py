import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

COLS_TO_DROP = ["num_outbound_cmds", "is_host_login", "is_guest_login"]

SKEWED_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "num_compromised", "num_root", "num_file_creations",
]

MACRO_CLASSES = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']

# Default SMOTE targets for minority classes (indices: Probe=2, R2L=3, U2R=4)
DEFAULT_SMOTE_STRATEGY = {2: 15000, 3: 20000, 4: 5000}


def make_label_encoder(y_train, y_val, y_test):
    """Fit LabelEncoder on train, transform all splits. Returns encoded arrays and fitted encoder."""
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    y_test_enc = encoder.transform(y_test)
    return encoder, y_train_enc, y_val_enc, y_test_enc
 

def drop_low_variance_cols(X_train, X_val, X_test):
    """Drop columns that carry no discriminative signal."""
    cols = [c for c in COLS_TO_DROP if c in X_train.columns]
    return X_train.drop(columns=cols), X_val.drop(columns=cols), X_test.drop(columns=cols)


def log_transform(X_train, X_val, X_test):
    """Apply log1p to skewed features to reduce distributional skew."""
    for col in SKEWED_FEATURES:
        if col in X_train.columns:
            X_train = X_train.copy()
            X_val = X_val.copy()
            X_test = X_test.copy()
            X_train[col] = np.log1p(X_train[col])
            X_val[col]   = np.log1p(X_val[col])
            X_test[col]  = np.log1p(X_test[col])
    return X_train, X_val, X_test


def scale(X_train, X_val, X_test):
    """Fit MinMaxScaler on train, apply to val and test. Returns scaled arrays and scaler."""
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def preprocess(X_train, X_val, X_test):
    """
    Full preprocessing pipeline:
      1. Drop low-variance columns
      2. Log-transform skewed features
      3. MinMax scale

    Returns (X_train, X_val, X_test, scaler) where X arrays are numpy arrays.
    """
    X_train, X_val, X_test = drop_low_variance_cols(X_train, X_val, X_test)
    X_train, X_val, X_test = log_transform(X_train, X_val, X_test)
    X_train, X_val, X_test, scaler = scale(X_train, X_val, X_test)
    return X_train, X_val, X_test, scaler

def scale_features(X_train, X_val, X_test):
    """
    MinMaxScaler fit on REAL training data only (before SMOTE),
    then applied to val and test. Returns scaled arrays and fitted scaler.
    """
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler

def make_calibration_split(X_val, y_val_enc, test_size=0.5, random_state=42):
    """Split validation set into val and calibration subsets."""
    X_calib, X_val_eval, y_calib_enc, y_val_eval_enc = train_test_split(X_val, y_val_enc, test_size=test_size, random_state=random_state, stratify=y_val_enc)
    return X_calib, X_val_eval, y_calib_enc, y_val_eval_enc

def apply_smote(X_train, y_train_enc, strategy=None, random_state=42):
    strategy = strategy or DEFAULT_SMOTE_STRATEGY
    smote = SMOTE(sampling_strategy=strategy, random_state=random_state)
    smote_enn = SMOTEENN(smote=smote, random_state=random_state)

    X_res, y_res = smote_enn.fit_resample(X_train, y_train_enc)
    return X_res, y_res

def build_features(X_train, X_val, X_test):
    X_train, X_val, X_test = drop_low_variance_cols(X_train, X_val, X_test)
    
    features = X_train.columns.tolist()

    X_train, X_val, X_test = log_transform(X_train, X_val, X_test)

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler, features