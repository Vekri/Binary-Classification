"""Export trained models as pickle/joblib artifacts for offline scoring."""

from __future__ import annotations

import io
import pickle
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd


SCORING_SNIPPET = '''\
"""Score new rows with a Binary Classification ML Pipeline artifact."""
import joblib
import pandas as pd

ARTIFACT = "scoring_model.joblib"  # or scoring_model.pkl

bundle = joblib.load(ARTIFACT)
model = bundle["model"]
feature_names = bundle["feature_names"]
label_map = bundle.get("label_map") or {}

# Load your scoring CSV (same raw columns used before modeling)
df = pd.read_csv("new_data.csv")

# Match training dummies / columns
X = pd.get_dummies(df, drop_first=True).fillna(0)
for col in feature_names:
    if col not in X.columns:
        X[col] = 0
X = X[feature_names]

proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
pred = model.predict(X)

# Optional: map encoded labels back
inv = {v: k for k, v in label_map.items()} if label_map else {}
labels = [inv.get(int(p), p) for p in pred]

out = df.copy()
out["prediction"] = labels
if proba is not None:
    out["probability"] = proba
out.to_csv("scored.csv", index=False)
print("Wrote scored.csv")
'''


def build_scoring_artifact(training_result: dict[str, Any]) -> dict[str, Any]:
    """Build a portable scoring package from train_best_model() output."""
    if not training_result or training_result.get("model") is None:
        raise ValueError("No trained model available to export.")

    return {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": training_result["model"],
        "model_name": training_result.get("model_name"),
        "feature_names": list(training_result.get("feature_names") or []),
        "label_map": training_result.get("label_map") or {},
        "metrics": training_result.get("metrics") or {},
        "usage": (
            "Load with joblib.load(...). Align columns to feature_names "
            "(pd.get_dummies + fill missing cols with 0), then model.predict / predict_proba."
        ),
    }


def artifact_to_joblib_bytes(artifact: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    joblib.dump(artifact, buf)
    return buf.getvalue()


def artifact_to_pickle_bytes(artifact: dict[str, Any]) -> bytes:
    return pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)


def model_only_pickle_bytes(training_result: dict[str, Any]) -> bytes:
    return pickle.dumps(training_result["model"], protocol=pickle.HIGHEST_PROTOCOL)


def feature_list_csv_bytes(feature_names: list[str]) -> bytes:
    df = pd.DataFrame({"feature_name": feature_names, "order": range(len(feature_names))})
    return df.to_csv(index=False).encode("utf-8")
