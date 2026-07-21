"""Automated feature engineering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.utils.helpers import get_column_types


def engineer_features(
    df: pd.DataFrame,
    target_col: str,
    options: dict[str, bool] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    log: dict[str, Any] = {"new_features": []}
    opts = options or {
        "interactions": True,
        "polynomial": True,
        "log_transform": True,
        "binning": True,
        "datetime_features": True,
    }
    types = get_column_types(result)
    numeric = [c for c in types["numeric"] if c != target_col]

    if opts.get("log_transform") and numeric:
        for col in numeric[:5]:
            if (result[col] > 0).all():
                new_col = f"log_{col}"
                result[new_col] = np.log1p(result[col])
                log["new_features"].append(new_col)

    if opts.get("polynomial") and len(numeric) >= 1:
        for col in numeric[:3]:
            new_col = f"{col}_squared"
            result[new_col] = result[col] ** 2
            log["new_features"].append(new_col)

    if opts.get("interactions") and len(numeric) >= 2:
        for i, c1 in enumerate(numeric[:3]):
            for c2 in numeric[i + 1 : 4]:
                new_col = f"{c1}_x_{c2}"
                result[new_col] = result[c1] * result[c2]
                log["new_features"].append(new_col)

    if opts.get("binning") and numeric:
        for col in numeric[:3]:
            try:
                new_col = f"{col}_binned"
                result[new_col] = pd.qcut(result[col], q=5, labels=False, duplicates="drop")
                log["new_features"].append(new_col)
            except ValueError:
                pass

    if opts.get("datetime_features") and types["datetime"]:
        for col in types["datetime"]:
            result[f"{col}_year"] = pd.to_datetime(result[col]).dt.year
            result[f"{col}_month"] = pd.to_datetime(result[col]).dt.month
            result[f"{col}_dayofweek"] = pd.to_datetime(result[col]).dt.dayofweek
            log["new_features"].extend(
                [f"{col}_year", f"{col}_month", f"{col}_dayofweek"]
            )

    return result, log
