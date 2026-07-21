"""Outlier detection using IQR and Z-score methods."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from app.utils.helpers import get_column_types


def detect_outliers_iqr(
    series: pd.Series, multiplier: float = 1.5
) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def detect_outliers_zscore(
    series: pd.Series, threshold: float = 3.0
) -> pd.Series:
    z = np.abs(stats.zscore(series.dropna()))
    mask = pd.Series(False, index=series.index)
    mask.loc[series.dropna().index] = z > threshold
    return mask


def outlier_summary(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    types = get_column_types(df)
    exclude = set(exclude_cols or [])
    rows = []
    for col in types["numeric"]:
        if col in exclude or df[col].nunique(dropna=True) <= 5:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        if method == "zscore":
            mask = detect_outliers_zscore(df[col], threshold)
        else:
            mask = detect_outliers_iqr(df[col], threshold)
        count = int(mask.sum())
        rows.append(
            {
                "column": col,
                "outlier_count": count,
                "outlier_pct": round(count / len(df) * 100, 2),
                "method": method,
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)


def outlier_boxplots(df: pd.DataFrame, max_cols: int = 6) -> go.Figure:
    numeric = df.select_dtypes(include="number").columns[:max_cols].tolist()
    if not numeric:
        fig = go.Figure()
        fig.add_annotation(text="No numeric columns", x=0.5, y=0.5, showarrow=False)
        return fig
    fig = go.Figure()
    for col in numeric:
        fig.add_trace(go.Box(y=df[col], name=col))
    fig.update_layout(title="Outlier Box Plots", height=450)
    return fig


def treat_outliers(
    df: pd.DataFrame,
    method: str = "cap",
    detection: str = "iqr",
    threshold: float = 1.5,
    columns: list[str] | None = None,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    types = get_column_types(result)
    exclude = set(exclude_cols or [])
    target_cols = [
        c for c in (columns or types["numeric"]) if c not in exclude
    ]
    # Skip near-binary / low-cardinality columns (e.g. the target)
    target_cols = [
        c
        for c in target_cols
        if result[c].nunique(dropna=True) > 5
    ]
    log: dict[str, Any] = {"actions": [], "outliers_treated": 0}

    for col in target_cols:
        if col not in result.columns:
            continue
        if detection == "zscore":
            mask = detect_outliers_zscore(result[col], threshold)
        else:
            mask = detect_outliers_iqr(result[col], threshold)
        count = int(mask.sum())
        if count == 0:
            continue

        if method == "remove":
            result = result[~mask]
            log["actions"].append(f"{col}: removed {count} rows")
        elif method == "cap":
            q1, q3 = result[col].quantile(0.25), result[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            result.loc[result[col] < lower, col] = lower
            result.loc[result[col] > upper, col] = upper
            log["actions"].append(f"{col}: capped {count} outliers")
        log["outliers_treated"] += count

    return result, log
