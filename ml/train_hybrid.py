"""
Phase-2 hybrid detection model for the RG26 Intelligent Defence System.

Phase 1 trained a single Random Forest on NSL-KDD and reached ~75% test
accuracy. The reason it was not higher is structural, not a tuning problem:
KDDTest+ contains 17 attack types that never appear in KDDTrain+, and a
supervised classifier has no class to put them in, so it calls them Normal.

Phase 2 therefore adds a second, unsupervised detector:

    Random Forest   - supervised, names the attack family it was trained on
    Isolation Forest- unsupervised, trained on normal traffic only, flags any
                      flow that does not look like normal behaviour

Decision rule used by the live system:

    RF says attack            -> threat, labelled with the RF family
    RF says normal + IF flags -> threat, labelled "Unknown / Zero-day"
    both say normal           -> normal

Run:  python ml/train_hybrid.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.preprocess import CLASSES, Preprocessor, load_nsl_kdd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "ml" / "artifacts"
TRAIN_FILE = ROOT / "KDDTrain+.txt"
TEST_FILE = ROOT / "KDDTest+.txt"

RANDOM_STATE = 42
IF_CONTAMINATION = 0.05  # share of normal training traffic the IF may flag


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def main() -> None:
    import joblib
    from imblearn.over_sampling import ADASYN
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score,
                                 precision_recall_fscore_support)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # -- 1. data ------------------------------------------------------------
    banner("1. Loading NSL-KDD")
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        sys.exit(f"Missing dataset. Expected {TRAIN_FILE} and {TEST_FILE}.")

    train = load_nsl_kdd(str(TRAIN_FILE))
    test = load_nsl_kdd(str(TEST_FILE))
    print(f"  train: {train.shape}   test: {test.shape}")

    unseen = sorted(set(test["label"]) - set(train["label"]))
    print(f"  attack types present only in the test set: {len(unseen)}")
    print(f"  -> {', '.join(unseen)}")

    # -- 2. preprocessing ---------------------------------------------------
    banner("2. Preprocessing (fit on train only, so there is no leakage)")
    pp = Preprocessor(n_features=40).fit(train)
    X_train = pp.transform(train.drop_duplicates())
    y_train = train.drop_duplicates()["category"].values
    X_test = pp.transform(test)
    y_test = test["category"].values

    # -- 3. class balancing -------------------------------------------------
    banner("3. Class balancing with ADASYN")
    print("  before:", pd.Series(y_train).value_counts().to_dict())
    X_bal, y_bal = ADASYN(random_state=RANDOM_STATE, n_neighbors=5).fit_resample(
        X_train, y_train
    )
    print("  after :", pd.Series(y_bal).value_counts().to_dict())

    # -- 4. supervised model ------------------------------------------------
    banner("4. Random Forest (supervised, known attack families)")
    rf = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight=None
    )
    rf.fit(X_bal, y_bal)
    rf_pred = rf.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_macro_f1 = f1_score(y_test, rf_pred, average="macro")
    print(f"  multiclass accuracy : {rf_acc * 100:.2f}%")
    print(f"  macro F1            : {rf_macro_f1 * 100:.2f}%")
    print(classification_report(y_test, rf_pred, digits=3, zero_division=0))

    # -- 5. unsupervised model ----------------------------------------------
    banner("5. Isolation Forest (unsupervised, trained on normal traffic only)")
    X_normal = X_train[y_train == "Normal"]
    print(f"  normal training flows: {len(X_normal)}")
    iso = IsolationForest(
        n_estimators=200,
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_normal)
    iso_flag = iso.predict(X_test) == -1  # True = anomalous

    # -- 6. hybrid decision rule -------------------------------------------
    banner("6. Hybrid decision rule")
    y_attack = y_test != "Normal"           # ground truth, attack vs normal
    rf_attack = rf_pred != "Normal"         # Phase-1 behaviour
    hybrid_attack = rf_attack | iso_flag    # Phase-2 behaviour

    hybrid_label = np.where(
        rf_attack, rf_pred, np.where(iso_flag, "Unknown/Zero-day", "Normal")
    )

    def detection_metrics(pred_attack: np.ndarray) -> dict:
        p, r, f, _ = precision_recall_fscore_support(
            y_attack, pred_attack, average="binary", zero_division=0
        )
        normal_mask = ~y_attack
        fpr = float(pred_attack[normal_mask].mean())
        return {
            "accuracy": float(accuracy_score(y_attack, pred_attack)),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "false_positive_rate": fpr,
        }

    rf_only = detection_metrics(rf_attack)
    hybrid = detection_metrics(hybrid_attack)

    print(f"  {'metric':<22}{'RF only':>12}{'Hybrid':>12}{'change':>12}")
    for key in ["accuracy", "precision", "recall", "f1", "false_positive_rate"]:
        a, b = rf_only[key] * 100, hybrid[key] * 100
        print(f"  {key:<22}{a:>11.2f}%{b:>11.2f}%{b - a:>+11.2f}")

    # Per-family recall: this is where the hybrid earns its place.
    banner("7. Per-family detection rate (attack caught at all, yes or no)")
    per_family = {}
    print(f"  {'family':<12}{'count':>8}{'RF only':>12}{'Hybrid':>12}")
    for fam in CLASSES[1:]:
        mask = y_test == fam
        if not mask.any():
            continue
        a = float(rf_attack[mask].mean())
        b = float(hybrid_attack[mask].mean())
        per_family[fam] = {"support": int(mask.sum()), "rf_only": a, "hybrid": b}
        print(f"  {fam:<12}{mask.sum():>8}{a * 100:>11.2f}%{b * 100:>11.2f}%")

    unseen_mask = test["label"].isin(unseen).values
    unseen_rf = float(rf_attack[unseen_mask].mean())
    unseen_hybrid = float(hybrid_attack[unseen_mask].mean())
    print(f"\n  unseen attack types only ({unseen_mask.sum()} flows):")
    print(f"    RF only : {unseen_rf * 100:.2f}%")
    print(f"    Hybrid  : {unseen_hybrid * 100:.2f}%")

    # -- 8. save ------------------------------------------------------------
    banner("8. Saving artifacts")
    joblib.dump(
        {"preprocessor": pp, "rf": rf, "iso": iso, "classes": CLASSES},
        ARTIFACTS / "hybrid_model.joblib",
    )

    importance = (
        pd.Series(rf.feature_importances_, index=pp.selected)
        .sort_values(ascending=False)
    )

    metrics = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "unseen_attack_types": unseen,
        },
        "random_forest_multiclass": {
            "accuracy": float(rf_acc),
            "macro_f1": float(rf_macro_f1),
        },
        "detection": {"rf_only": rf_only, "hybrid": hybrid},
        "per_family": per_family,
        "unseen_attacks": {
            "flows": int(unseen_mask.sum()),
            "rf_only": unseen_rf,
            "hybrid": unseen_hybrid,
        },
        "top_features": importance.head(15).round(4).to_dict(),
        "isolation_forest_contamination": IF_CONTAMINATION,
        "training_seconds": round(time.time() - started, 1),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  {ARTIFACTS / 'hybrid_model.joblib'}")
    print(f"  {ARTIFACTS / 'metrics.json'}")

    _save_plots(y_test, hybrid_label, rf_only, hybrid, per_family, importance)
    print(f"\nDone in {metrics['training_seconds']}s.")


def _save_plots(y_test, hybrid_label, rf_only, hybrid, per_family, importance) -> None:
    """Figures for the presentation. Failure here must not lose the model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    except Exception as exc:  # pragma: no cover
        print(f"  (plots skipped: {exc})")
        return

    blue = "#0037A4"

    labels = CLASSES + ["Unknown/Zero-day"]
    cm = confusion_matrix(y_test, hybrid_label, labels=labels)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        cmap="Blues", values_format="d", ax=ax, colorbar=True
    )
    ax.set_title("Hybrid Model - Confusion Matrix (NSL-KDD Test)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(ARTIFACTS / "hybrid_confusion_matrix.png", dpi=150)
    plt.close(fig)

    keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - 0.2, [rf_only[k] * 100 for k in keys], 0.4, label="RF only (Phase 1)",
           color="#9DB4E8")
    ax.bar(x + 0.2, [hybrid[k] * 100 for k in keys], 0.4, label="Hybrid (Phase 2)",
           color=blue)
    for i, k in enumerate(keys):
        ax.text(i - 0.2, rf_only[k] * 100, f"{rf_only[k] * 100:.1f}", ha="center",
                va="bottom", fontsize=9)
        ax.text(i + 0.2, hybrid[k] * 100, f"{hybrid[k] * 100:.1f}", ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(x, [k.capitalize() for k in keys])
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Attack Detection: Random Forest vs Hybrid")
    ax.legend()
    plt.tight_layout()
    fig.savefig(ARTIFACTS / "detection_comparison.png", dpi=150)
    plt.close(fig)

    fams = list(per_family)
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - 0.2, [per_family[f]["rf_only"] * 100 for f in fams], 0.4,
           label="RF only", color="#9DB4E8")
    ax.bar(x + 0.2, [per_family[f]["hybrid"] * 100 for f in fams], 0.4,
           label="Hybrid", color=blue)
    ax.set_xticks(x, fams)
    ax.set_ylabel("Detection rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-family Detection Rate")
    ax.legend()
    plt.tight_layout()
    fig.savefig(ARTIFACTS / "per_family_detection.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    importance.head(15).sort_values().plot(kind="barh", color=blue, ax=ax)
    ax.set_title("Top 15 Features (Random Forest importance)")
    plt.tight_layout()
    fig.savefig(ARTIFACTS / "rf_feature_importance.png", dpi=150)
    plt.close(fig)

    print("  4 figures written to ml/artifacts/")


if __name__ == "__main__":
    main()
