"""Shared helpers for the ML pipeline."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split


def split_features_target(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def get_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_cols,
    }


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def fig_to_bytes(fig: go.Figure) -> bytes:
    return fig.to_image(format="png", engine="kaleido")


def dataframe_profile_summary(df: pd.DataFrame) -> dict[str, Any]:
    types = get_column_types(df)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "numeric_cols": len(types["numeric"]),
        "categorical_cols": len(types["categorical"]),
        "datetime_cols": len(types["datetime"]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(df.isna().sum().sum() / df.size * 100, 2),
    }


def create_missing_heatmap(df: pd.DataFrame) -> go.Figure:
    missing = df.isna().astype(int)
    if missing.sum().sum() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values detected",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(title="Missing Values Heatmap", height=400)
        return fig

    sample = missing.head(min(500, len(missing)))
    fig = px.imshow(
        sample.T,
        labels=dict(x="Row", y="Column", color="Missing"),
        color_continuous_scale="Reds",
        aspect="auto",
        title="Missing Values Heatmap (sample)",
    )
    fig.update_layout(height=400 + len(df.columns) * 8)
    return fig


def safe_encode_target(y: pd.Series) -> tuple[pd.Series, dict]:
    if y.dtype == "object" or str(y.dtype) == "category":
        classes = sorted(y.dropna().unique(), key=str)
        mapping = {cls: idx for idx, cls in enumerate(classes)}
        return y.map(mapping), mapping
    return y, {}
