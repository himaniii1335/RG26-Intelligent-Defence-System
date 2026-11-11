"""
Inference wrapper around the trained hybrid model.

The backend holds one HybridDetector instance and calls `predict` for every
network flow that arrives. Nothing here retrains or refits: the preprocessing
state was fitted once on the training set and travels inside the model file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "hybrid_model.joblib"


class HybridDetector:
    """Random Forest (known families) + Isolation Forest (everything else)."""

    def __init__(self, artifact: Path | str = ARTIFACT):
        import joblib

        artifact = Path(artifact)
        if not artifact.exists():
            raise FileNotFoundError(
                f"Model not found at {artifact}. Run: python ml/train_hybrid.py"
            )
        bundle = joblib.load(artifact)
        self.pp = bundle["preprocessor"]
        self.rf = bundle["rf"]
        self.iso = bundle["iso"]
        self.classes = list(self.rf.classes_)
        self.feature_names = list(self.pp.selected)

    def predict(self, flow: dict) -> dict:
        """Score one raw NSL-KDD flow.

        Returns the verdict plus the evidence the dashboard shows when an
        analyst clicks an alert. An alert nobody can explain is not actionable,
        which is the gap the explainability papers in our review point at.
        """
        X = self.pp.transform_one(flow)

        rf_label = str(self.rf.predict(X)[0])
        proba = self.rf.predict_proba(X)[0]
        rf_confidence = float(proba.max())

        # decision_function: negative means "outlier" for IsolationForest
        iso_score = float(self.iso.decision_function(X)[0])
        iso_flag = bool(self.iso.predict(X)[0] == -1)

        if rf_label != "Normal":
            label, source, confidence = rf_label, "signature (Random Forest)", rf_confidence
        elif iso_flag:
            label, source = "Unknown/Zero-day", "anomaly (Isolation Forest)"
            # decision_function is (score - offset), so it is small and negative
            # for outliers; 0.15 is about the deepest value seen on this data,
            # which makes the ratio a readable 0-1 anomaly strength rather than
            # a probability. It is reported as strength, never as confidence.
            confidence = float(min(1.0, max(0.0, -iso_score / 0.15)))
        else:
            label, source, confidence = "Normal", "both detectors agree", rf_confidence

        return {
            "label": label,
            "is_threat": label != "Normal",
            "source": source,
            "confidence": round(confidence, 3),
            "rf_label": rf_label,
            "rf_probabilities": {
                c: round(float(p), 3) for c, p in zip(self.rf.classes_, proba)
            },
            "anomaly_score": round(iso_score, 4),
            "anomaly_flag": iso_flag,
            "top_features": self._top_features(X),
        }

    def _top_features(self, X, k: int = 5) -> list[dict]:
        """The features that drove this particular decision.

        Global RF importance weighted by how far this flow's value sits from
        zero after scaling. This is a cheap stand-in for SHAP, which is the
        planned Phase-3 upgrade; it is honest about being an approximation.
        """
        values = X.iloc[0].to_numpy()
        contribution = self.rf.feature_importances_ * np.abs(values)
        order = np.argsort(contribution)[::-1][:k]
        return [
            {
                "feature": self.feature_names[i],
                "value": round(float(values[i]), 4),
                "weight": round(float(contribution[i]), 4),
            }
            for i in order
        ]
