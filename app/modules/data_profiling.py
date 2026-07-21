"""Data profiling using ydata-profiling and custom summaries."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.utils.helpers import create_missing_heatmap, dataframe_profile_summary


def generate_profile_report(df: pd.DataFrame, minimal: bool = True):
    from ydata_profiling import ProfileReport

    return ProfileReport(
        df,
        title="Data Profile Report",
        explorative=True,
        minimal=minimal,
        progress_bar=False,
    )


def profile_to_html(report) -> str:
    return report.to_html()


def get_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for col in df.columns:
        series = df[col]
        entry: dict[str, Any] = {
            "column": col,
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            entry.update(
                {
                    "mean": round(series.mean(), 4) if series.notna().any() else None,
                    "std": round(series.std(), 4) if series.notna().any() else None,
                    "min": series.min() if series.notna().any() else None,
                    "max": series.max() if series.notna().any() else None,
                    "skew": round(series.skew(), 4) if series.notna().any() else None,
                }
            )
        else:
            mode = series.mode()
            entry["top_value"] = mode.iloc[0] if len(mode) else None
            entry["top_freq"] = (
                int(series.value_counts().iloc[0]) if series.notna().any() else 0
            )
        stats.append(entry)
    return pd.DataFrame(stats)


def correlation_matrix(df: pd.DataFrame) -> go.Figure:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 numeric columns",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig
    corr = numeric.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Matrix",
    )
    fig.update_layout(height=500)
    return fig


def distribution_plots(df: pd.DataFrame, max_cols: int = 6) -> list[go.Figure]:
    figures = []
    numeric = df.select_dtypes(include="number").columns[:max_cols]
    for col in numeric:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution: {col}")
        fig.update_layout(height=350)
        figures.append(fig)
    return figures


def run_profiling(df: pd.DataFrame) -> dict:
    summary = dataframe_profile_summary(df)
    stats_df = get_basic_stats(df)
    return {
        "summary": summary,
        "stats": stats_df,
        "missing_heatmap": create_missing_heatmap(df),
        "correlation": correlation_matrix(df),
        "distributions": distribution_plots(df),
    }
