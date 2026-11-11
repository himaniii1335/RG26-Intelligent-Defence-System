"""
NSL-KDD preprocessing for the RG26 Intelligent Defence System.

This is the Phase-1 notebook pipeline refactored into one reusable module so
that training and the live backend use exactly the same transformation.
Any difference between the two would make the live predictions meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------------
# Dataset schema (NSL-KDD has no header row, so the column names are fixed here)
# ----------------------------------------------------------------------------

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
COLS = FEATURES + ["label", "difficulty"]

CATEGORICAL = ["protocol_type", "service", "flag"]
NON_FEATURE = ["label", "difficulty", "category"]

# The 39 raw attack labels collapse into the four standard NSL-KDD families.
DOS = ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "mailbomb",
       "processtable", "udpstorm"]
PROBE = ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"]
R2L = ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient",
       "warezmaster", "sendmail", "named", "snmpgetattack", "snmpguess", "xlock",
       "xsnoop", "worm"]
U2R = ["buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel", "ps",
       "sqlattack", "xterm"]

CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def to_category(label: str) -> str:
    """Map a raw NSL-KDD attack label to its family."""
    if label == "normal":
        return "Normal"
    if label in DOS:
        return "DoS"
    if label in PROBE:
        return "Probe"
    if label in R2L:
        return "R2L"
    if label in U2R:
        return "U2R"
    return "Unknown"


def load_nsl_kdd(path: str) -> pd.DataFrame:
    """Read a KDDTrain+/KDDTest+ text file and attach the category column."""
    df = pd.read_csv(path, names=COLS)
    df["category"] = df["label"].apply(to_category)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Domain feature engineering (identical to the Phase-1 notebook).

    Raw byte counts span several orders of magnitude, so they are log-compressed;
    the derived ratios and sums encode the behaviour that separates attack
    families (volume imbalance for DoS, error rates for scans, privilege
    activity for U2R/R2L).
    """
    df = df.copy()
    df["log_src_bytes"] = np.log1p(df["src_bytes"])
    df["log_dst_bytes"] = np.log1p(df["dst_bytes"])
    df["bytes_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
    df["total_error_rate"] = df["serror_rate"] + df["rerror_rate"]
    df["privilege_activity"] = (
        df["root_shell"] + df["su_attempted"] + df["num_root"] + df["num_shells"]
    )
    return df.drop(columns=["src_bytes", "dst_bytes"])


class Preprocessor:
    """Fit-once / transform-many preprocessing state.

    Everything learned from the training set (which columns survive, the
    one-hot vocabulary, the scaler ranges, the selected features) is stored on
    this object and pickled alongside the models, so a single live event is
    transformed exactly as the training rows were.
    """

    def __init__(self, n_features: int = 40):
        self.n_features = n_features
        self.constant_cols: list[str] = []
        self.dummy_cols: list[str] = []
        self.selected: list[str] = []
        self.scaler: MinMaxScaler | None = None

    # -- fitting -------------------------------------------------------------

    def fit(self, train: pd.DataFrame, verbose: bool = True) -> "Preprocessor":
        train = train.drop_duplicates()

        numeric = train[FEATURES].select_dtypes(include=np.number).columns
        self.constant_cols = [c for c in numeric if train[c].nunique() <= 1]

        fe = engineer(train.drop(columns=self.constant_cols))
        X = pd.get_dummies(fe.drop(columns=NON_FEATURE), columns=CATEGORICAL)
        self.dummy_cols = list(X.columns)

        self.scaler = MinMaxScaler().fit(X)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=self.dummy_cols)

        # Drop one of every near-duplicate pair, then keep the most informative.
        corr = Xs.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        redundant = [c for c in upper.columns if (upper[c] > 0.95).any()]
        Xs = Xs.drop(columns=redundant)

        mi = pd.Series(
            mutual_info_classif(Xs, train["category"].values, random_state=42),
            index=Xs.columns,
        ).sort_values(ascending=False)
        self.selected = list(mi.head(self.n_features).index)
        self.mutual_info = mi

        if verbose:
            print(f"  constant columns dropped : {self.constant_cols}")
            print(f"  one-hot feature space    : {len(self.dummy_cols)}")
            print(f"  redundant (corr > 0.95)  : {len(redundant)}")
            print(f"  features selected        : {len(self.selected)}")
        return self

    # -- applying ------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform any frame with the raw NSL-KDD columns into model input."""
        df = df.drop(columns=[c for c in self.constant_cols if c in df.columns])
        fe = engineer(df)
        fe = fe.drop(columns=[c for c in NON_FEATURE if c in fe.columns])
        X = pd.get_dummies(fe, columns=CATEGORICAL)
        # Unseen service/flag values simply produce all-zero columns here, which
        # is what lets the backend score live events it has never seen before.
        X = X.reindex(columns=self.dummy_cols, fill_value=0)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=self.dummy_cols)
        return Xs[self.selected]

    def transform_one(self, event: dict) -> pd.DataFrame:
        """Transform a single raw flow supplied as a dictionary."""
        row = {c: event.get(c, 0) for c in FEATURES}
        return self.transform(pd.DataFrame([row]))
