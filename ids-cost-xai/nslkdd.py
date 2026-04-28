import os
import urllib.request
from numpy.random import f
import pandas as pd
from logger import Logger
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

log = Logger()

# https://www.kaggle.com/datasets/hassan06/nslkdd

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

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
CRITICAL_FEATURES = ["root_shell", "su_attempted", "num_root", "num_failed_logins", "num_compromised", "num_shells"]

MACRO_CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
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
    "rootkit": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R"
    }

URLS = {
    "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "test": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
    }


def download_data(dir: str = "data/nslkdd/raw") -> dict[str, str]:
    os.makedirs(dir, exist_ok=True)
    paths = {}
    for split, url in URLS.items():
        destination = os.path.join(dir, f"nslkdd_{split}.txt")

        if not os.path.exists(destination):
            log.info(f"[NSL-KDD] Downloading {split} data...")
            urllib.request.urlretrieve(url, destination)
            log.info(f"[NSL-KDD] Downloading {split} data completed.")
        else:
            log.info(f"[NSL-KDD] {split} data already exists. Skipping download.")
        paths[split] = destination

    return paths

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=NSLKDD_COLUMNS)
    df["label"] = df["label"].apply(lambda x: x.split(".")[0])  # Remove trailing dot
    return df

def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["macro_label"] = df["label"].apply(lambda x: ATTACK_MAPPING.get(x, "Unknown"))
    return df

def audit_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    class_counts = df["macro_label"].value_counts()
    pct = (class_counts / len(df) * 100).round(2)
    audit = pd.concat([class_counts, pct], axis=1, keys=["count", "percentage"])
    audit.index.name = "class"
    log.info("Imbalance audit: ")
    log.blank(audit.to_string())
    log.info(f"Total samples: {len(df)}")
    majority = audit["count"].max()
    log.info("Imbalance ratio (majority/minority):")
    for cls, count in class_counts.items():
        if count > 0:
            ratio = majority / count
            log.info(f"  {cls:8s}: {ratio:>8.1f}x")

    return audit


def encode_categorical(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    encoders = {}

    train_enc = train_df.copy()
    test_enc = test_df.copy()

    for feature in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        train_enc[feature] = le.fit_transform(train_enc[feature].astype(str))
        test_enc[feature] = test_df[feature].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        encoders[feature] = le
    
    return train_enc, test_enc, encoders

def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, features: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[features] = scaler.fit_transform(X_train[features])
    X_val_scaled[features] = scaler.transform(X_val[features])
    X_test_scaled[features] = scaler.transform(X_test[features])
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def run_pipeline(val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42):

    # Download and load data
    paths = download_data()
    train_df = load_raw_data(paths["train"])
    test_df = load_raw_data(paths["test"])
    log.success("Data loading completed.")
    
    # Map attack labels to macro classes
    train_df = map_labels(train_df)
    test_df = map_labels(test_df)
    log.success("Label mapping completed.")

    # Audit class imbalance
    audit_train = audit_imbalance(train_df)
    audit_test = audit_imbalance(test_df)
    log.success("Imbalance audit completed.")

    # Encode categorical features
    train_enc, test_enc, encoders = encode_categorical(train_df, test_df)
    log.success("Categorical feature encoding completed.")

    # Prepare train/val/test splits
    drop_cols = ["label", "difficulty_level", "macro_label"]
    features = [col for col in train_enc.columns if col not in drop_cols]
    x_trainval = train_enc[features]
    y_trainval = train_enc["macro_label"]

    # Calculate relative validation size based on desired test size
    relative_val = val_size / (1.0 - test_size)

    # Stratified split to maintain class distribution
    X_train, X_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=relative_val, random_state=random_state, stratify=y_trainval)

    X_test = test_enc[features]
    y_test = test_enc["macro_label"]

    log.success(f"Data splitting completed. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test, features)

    for name, X, y in [("Train", X_train, y_train), ("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        out = X.copy()
        out["macro_label"] = y.values
        save_dir = "data/nslkdd/processed"
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"nslkdd_{name.lower()}.csv")
        out.to_csv(path, index=False)

    log.success("Data scaling and saving completed.")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler, "encoders": encoders, 
        "audit_train": audit_train, "audit_test": audit_test
        }