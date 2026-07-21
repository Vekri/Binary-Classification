"""Data quality scoring framework."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.utils.helpers import get_column_types


def _completeness_score(df: pd.DataFrame) -> float:
    return (1 - df.isna().sum().sum() / df.size) * 100


def _uniqueness_score(df: pd.DataFrame) -> float:
    dup_ratio = df.duplicated().sum() / len(df)
    return (1 - dup_ratio) * 100


def _validity_score(df: pd.DataFrame) -> float:
    issues = 0
    total_checks = 0
    types = get_column_types(df)
    for col in types["numeric"]:
        total_checks += 1
        series = df[col].dropna()
        if len(series) and np.isinf(series).any():
            issues += 1
    for col in df.columns:
        total_checks += 1
        if df[col].astype(str).str.strip().eq("").any():
            issues += 1
    if total_checks == 0:
        return 100.0
    return (1 - issues / total_checks) * 100


def _consistency_score(df: pd.DataFrame) -> float:
    types = get_column_types(df)
    issues = 0
    checks = max(len(types["numeric"]), 1)
    for col in types["numeric"]:
        series = df[col].dropna()
        if len(series) > 1:
            cv = series.std() / (abs(series.mean()) + 1e-9)
            if cv > 10:
                issues += 1
    return (1 - issues / checks) * 100


def compute_quality_score(df: pd.DataFrame) -> dict[str, Any]:
    completeness = _completeness_score(df)
    uniqueness = _uniqueness_score(df)
    validity = _validity_score(df)
    consistency = _consistency_score(df)

    weights = {
        "completeness": 0.35,
        "uniqueness": 0.20,
        "validity": 0.25,
        "consistency": 0.20,
    }
    overall = (
        completeness * weights["completeness"]
        + uniqueness * weights["uniqueness"]
        + validity * weights["validity"]
        + consistency * weights["consistency"]
    )

    grade = (
        "A" if overall >= 90
        else "B" if overall >= 80
        else "C" if overall >= 70
        else "D" if overall >= 60
        else "F"
    )

    return {
        "overall_score": round(overall, 1),
        "grade": grade,
        "dimensions": {
            "completeness": round(completeness, 1),
            "uniqueness": round(uniqueness, 1),
            "validity": round(validity, 1),
            "consistency": round(consistency, 1),
        },
        "weights": weights,
    }


def quality_radar_chart(scores: dict[str, Any]) -> go.Figure:
    dims = scores["dimensions"]
    categories = list(dims.keys())
    values = list(dims.values())

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Quality Score",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"Data Quality Score: {scores['overall_score']} ({scores['grade']})",
        height=450,
    )
    return fig


def column_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        null_pct = series.isna().mean() * 100
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        rows.append(
            {
                "column": col,
                "completeness": round(100 - null_pct, 1),
                "uniqueness": round(unique_ratio * 100, 1),
                "dtype": str(series.dtype),
            }
        )
    return pd.DataFrame(rows)
