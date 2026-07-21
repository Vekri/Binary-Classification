"""Duplicate row detection and handling."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px


def duplicate_summary(df: pd.DataFrame) -> dict[str, Any]:
    total_dups = int(df.duplicated().sum())
    total_dups_keep_first = int(df.duplicated(keep="first").sum())
    exact_pct = round(total_dups / len(df) * 100, 2) if len(df) else 0

    col_dups = []
    for col in df.columns:
        dup_count = int(df[col].duplicated().sum())
        if dup_count > 0:
            col_dups.append({"column": col, "duplicate_values": dup_count})

    return {
        "total_duplicate_rows": total_dups,
        "duplicate_pct": exact_pct,
        "unique_rows": len(df) - total_dups_keep_first,
        "column_duplicates": pd.DataFrame(col_dups) if col_dups else pd.DataFrame(),
    }


def find_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    mask = df.duplicated(subset=subset, keep=False)
    return df[mask].sort_values(by=list(subset or df.columns))


def remove_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = len(df)
    result = df.drop_duplicates(subset=subset, keep=keep)
    after = len(result)
    return result, {
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
        "keep": keep,
        "subset": subset or "all columns",
    }


def duplicate_chart(df: pd.DataFrame):
    summary = duplicate_summary(df)
    fig = px.pie(
        values=[summary["unique_rows"], summary["total_duplicate_rows"]],
        names=["Unique Rows", "Duplicate Rows"],
        title="Row Uniqueness",
    )
    fig.update_layout(height=400)
    return fig
