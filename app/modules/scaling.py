"""Scaling recommendations and application."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from app.utils.helpers import get_column_types


SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
}


def recommend_scaling(df: pd.DataFrame) -> pd.DataFrame:
    types = get_column_types(df)
    rows = []
    for col in types["numeric"]:
        series = df[col].dropna()
        if len(series) < 2:
            continue
        skew = abs(series.skew())
        has_outliers = (
            (series < series.quantile(0.25) - 1.5 * (series.quantile(0.75) - series.quantile(0.25)))
            | (series > series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25)))
        ).sum() > 0

        if has_outliers or skew > 2:
            scaler = "robust"
            reason = "Outliers or heavy skew — robust scaling"
        elif series.min() >= 0 and series.max() <= 1:
            scaler = "none"
            reason = "Already normalized [0,1] — no scaling needed"
        elif skew > 0.5:
            scaler = "minmax"
            reason = "Moderate skew — min-max scaling"
        else:
            scaler = "standard"
            reason = "Approximately normal — standard scaling"
        rows.append(
            {
                "column": col,
                "skewness": round(skew, 3),
                "has_outliers": has_outliers,
                "recommended_scaler": scaler,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def apply_scaling(
    df: pd.DataFrame,
    target_col: str,
    scalers: dict[str, str] | None = None,
    global_scaler: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    log: dict[str, Any] = {"applied": []}
    types = get_column_types(result)

    if global_scaler and global_scaler != "none":
        numeric_cols = [c for c in types["numeric"] if c != target_col]
        scaler_cls = SCALERS[global_scaler]
        scaler = scaler_cls()
        result[numeric_cols] = scaler.fit_transform(result[numeric_cols])
        log["applied"].append(f"Global {global_scaler} on {numeric_cols}")
        return result, log

    if scalers is None:
        rec = recommend_scaling(result)
        scalers = {
            row["column"]: row["recommended_scaler"] for _, row in rec.iterrows()
        }

    for col, method in scalers.items():
        if col == target_col or col not in result.columns or method == "none":
            continue
        scaler_cls = SCALERS.get(method)
        if scaler_cls:
            scaler = scaler_cls()
            result[[col]] = scaler.fit_transform(result[[col]])
            log["applied"].append(f"{col}: {method}")

    return result, log
