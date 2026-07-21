"""Missing value analysis and treatment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.impute import KNNImputer, SimpleImputer

from app.utils.helpers import get_column_types


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing_count": null_count,
                "missing_pct": round(null_count / len(df) * 100, 2),
                "dtype": str(df[col].dtype),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


def recommend_missing_strategy(df: pd.DataFrame) -> pd.DataFrame:
    recommendations = []
    types = get_column_types(df)
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct == 0:
            continue
        dtype = df[col].dtype
        if pct > 50:
            strategy = "drop_column"
            reason = "Over 50% missing — consider dropping"
        elif col in types["numeric"]:
            skew = df[col].skew() if df[col].notna().any() else 0
            if abs(skew) > 1:
                strategy = "median"
                reason = "Skewed numeric — median is robust"
            else:
                strategy = "mean"
                reason = "Symmetric numeric — mean imputation"
        elif col in types["categorical"]:
            strategy = "mode"
            reason = "Categorical — use most frequent value"
        else:
            strategy = "mode"
            reason = "Default to mode imputation"
        recommendations.append(
            {
                "column": col,
                "missing_pct": round(pct, 2),
                "recommended_strategy": strategy,
                "reason": reason,
            }
        )
    return pd.DataFrame(recommendations)


def missing_bar_chart(df: pd.DataFrame):
    summary = missing_value_summary(df)
    summary = summary[summary["missing_count"] > 0]
    if summary.empty:
        return None
    fig = px.bar(
        summary,
        x="column",
        y="missing_pct",
        title="Missing Values by Column (%)",
        labels={"missing_pct": "Missing %"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=400)
    return fig


def apply_missing_treatment(
    df: pd.DataFrame,
    strategies: dict[str, str] | None = None,
    default_numeric: str = "median",
    default_categorical: str = "mode",
    drop_threshold: float = 0.5,
    use_knn: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    log: dict[str, Any] = {"actions": [], "dropped_columns": []}
    types = get_column_types(result)

    if strategies is None:
        rec = recommend_missing_strategy(result)
        strategies = {
            row["column"]: row["recommended_strategy"]
            for _, row in rec.iterrows()
        }

    cols_to_drop = [
        col
        for col, strat in strategies.items()
        if strat == "drop_column" or result[col].isna().mean() > drop_threshold
    ]
    if cols_to_drop:
        result = result.drop(columns=cols_to_drop)
        log["dropped_columns"] = cols_to_drop
        log["actions"].append(f"Dropped columns: {cols_to_drop}")

    numeric_cols = [c for c in types["numeric"] if c in result.columns and result[c].isna().any()]
    cat_cols = [
        c
        for c in types["categorical"]
        if c in result.columns and result[c].isna().any()
    ]

    if use_knn and numeric_cols:
        imputer = KNNImputer(n_neighbors=5)
        result[numeric_cols] = imputer.fit_transform(result[numeric_cols])
        log["actions"].append(f"KNN imputation on: {numeric_cols}")
    else:
        for col in numeric_cols:
            strat = strategies.get(col, default_numeric)
            if strat in ("mean", "median", "most_frequent"):
                imp = SimpleImputer(strategy=strat if strat != "mode" else "most_frequent")
                result[[col]] = imp.fit_transform(result[[col]])
                log["actions"].append(f"{col}: {strat} imputation")

    for col in cat_cols:
        mode_val = result[col].mode()
        fill_val = mode_val.iloc[0] if len(mode_val) else "Unknown"
        result[col] = result[col].fillna(fill_val)
        log["actions"].append(f"{col}: mode imputation ({fill_val})")

    log["remaining_missing"] = int(result.isna().sum().sum())
    return result, log
