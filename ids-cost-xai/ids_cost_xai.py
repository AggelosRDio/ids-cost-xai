import os
import urllib
from logger import Logger

log = Logger()

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

URLS = {
    "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "test": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
    }


def download_data(dir: str = "data/raw") -> dict[str, str]:
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