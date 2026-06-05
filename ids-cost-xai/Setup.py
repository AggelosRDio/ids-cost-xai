import os
import sys
import urllib.request

import numpy as np
import pandas as pd

from logger import Logger
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

log = Logger()

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

NSLKDD_URLS = {
    "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "test":  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
}

NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty_level",
]

ATTACK_MAPPING = {
    # Normal
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS", "mailbomb": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

COLS_TO_DROP = ["num_outbound_cmds", "is_host_login", "is_guest_login"]

SKEWED_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "num_compromised", "num_root",
    "num_file_creations",
]

UNSW_KAGGLE_DATASET = "mrwellsdavid/unsw-nb15"
UNSW_KAGGLE_FILES   = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]

UNSW_SEVERITY_MAPPING = {
    "Normal":         "Normal",
    "Fuzzers":        "Probe",
    "Analysis":       "Probe",
    "Backdoors":      "R2L",
    "Dos":            "DoS",
    "Exploits":       "R2L",
    "Generic":        "DoS",
    "Reconnaissance": "Probe",
    "Shellcode":      "U2R",
    "Worms":          "DoS",
}

UNSW_DROP_COLS   = ["id", "srcip", "sport", "dstip", "dsport",
                    "Stime", "Ltime", "attack_cat", "Label"]
UNSW_CATEGORICAL = ["proto", "service", "state"]


# SHARED HELPERS
def _already_exists(*paths):
    return all(os.path.exists(p) for p in paths)


def _encode_categoricals(train_df, test_df, categorical_cols):
    encoders = {}
    train_enc, test_enc = train_df.copy(), test_df.copy()

    for col in categorical_cols:
        if col not in train_df.columns:
            continue
        encoder = LabelEncoder()
        train_enc[col] = encoder.fit_transform(train_df[col].astype(str))
        test_enc[col]  = test_df[col].astype(str).map(
            lambda x, enc=encoder: enc.transform([x])[0] if x in enc.classes_ else -1
        )
        encoders[col] = encoder

    return train_enc, test_enc, encoders


def _apply_transforms(X_train, X_val, X_test):
    """Drop low-variance cols, log-transform skewed features, and MinMax scale."""
    drop = [c for c in COLS_TO_DROP if c in X_train.columns]
    X_train = X_train.drop(columns=drop)
    X_val = X_val.drop(columns=drop)
    X_test = X_test.drop(columns=drop)

    for col in SKEWED_FEATURES:
        if col in X_train.columns:
            X_train[col] = np.log1p(X_train[col])
            X_val[col] = np.log1p(X_val[col])
            X_test[col] = np.log1p(X_test[col])

    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_s   = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_s, X_val_s, X_test_s


def _save_splits(tag, X_train, X_val, X_test, y_train, y_val, y_test):
    for split, X, y in [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]:
        out = X.copy()
        out["macro_label"] = y.values if hasattr(y, "values") else y
        path = os.path.join(PROCESSED_DIR, f"{tag}_{split}.csv")
        out.to_csv(path, index=False)
        log.info(f"Saved {split} split in {path}. Shape: {out.shape}")



# NSL-KDD activity functions
def _download_nslkdd(train_path, test_path):
    """Download raw NSL-KDD train/test files if not already on disk."""
    for split, url in NSLKDD_URLS.items():
        path = train_path if split == "train" else test_path
        if os.path.exists(path):
            log.info(f"[NSL-KDD] {split} file already exists, skipping download.")
            continue
        log.info(f"[NSL-KDD] Downloading {split} from {url} ...")
        try:
            urllib.request.urlretrieve(url, path)
            log.success(f"[NSL-KDD] Downloaded {split} in {path}")
        except Exception as e:
            log.error(f"[NSL-KDD] Failed to download {split}: {e}")
            raise


def _load_and_label_nslkdd(train_path, test_path):
    """
    Load raw text files, strip trailing dots from labels,
    and map fine-grained attack types to macro classes.
    Returns (train_df, test_df).
    """
    train_df = pd.read_csv(train_path, header=None, names=NSLKDD_COLUMNS)
    test_df  = pd.read_csv(test_path,  header=None, names=NSLKDD_COLUMNS)

    for df in (train_df, test_df):
        df["label"] = df["label"].str.rstrip(".")
        df["macro_label"] = df["label"].str.lower().map(ATTACK_MAPPING).fillna("Unknown")

    log.success("[NSL-KDD] Raw data loaded and labels mapped.")
    return train_df, test_df


def _build_nslkdd_splits(train_df, test_df):
    """
    Encode categoricals, stratified train/val split, apply transforms.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    drop_cols = ["label", "difficulty_level", "macro_label"]
    feat_cols = [c for c in train_df.columns if c not in drop_cols]

    train_enc, test_enc, _ = _encode_categoricals(
        train_df[feat_cols + ["macro_label"]],
        test_df[feat_cols  + ["macro_label"]],
        CATEGORICAL_FEATURES
    )

    X_tv, y_tv = train_enc[feat_cols], train_enc["macro_label"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=0.2, random_state=42, stratify=y_tv
    )

    X_te, y_te = test_enc[feat_cols], test_enc["macro_label"]

    X_tr, X_val, X_te = _apply_transforms(X_tr, X_val, X_te)

    log.success(f"[NSL-KDD] Splits ready — train: {len(X_tr)}, val: {len(X_val)}, test: {len(X_te)}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# NSL-KDD orchestrator
def setup_nslkdd():
    train_path = os.path.join(RAW_DIR, "nslkdd_train.txt")
    test_path  = os.path.join(RAW_DIR, "nslkdd_test.txt")
    out_paths  = [os.path.join(PROCESSED_DIR, f"nslkdd_{s}.csv") for s in ("train", "validation", "test")]

    if _already_exists(*out_paths):
        log.info("[NSL-KDD] Processed files already exist. Skipping setup.")
        return True

    _download_nslkdd(train_path, test_path)

    train_df, test_df = _load_and_label_nslkdd(train_path, test_path)

    X_tr, X_val, X_te, y_tr, y_val, y_te = _build_nslkdd_splits(train_df, test_df)

    _save_splits("nslkdd", X_tr, X_val, X_te, y_tr, y_val, y_te)

    log.success("[NSL-KDD] Setup complete.")
    return True

if __name__ == "__main__":

    nslkdd_ok = setup_nslkdd()
    # TODO: add UNSW setup and call here as well, with appropriate exit codes/logging.

    sys.exit(0 if nslkdd_ok else 1)
    
