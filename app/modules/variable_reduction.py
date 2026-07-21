"""Variable reduction via PCA and variance threshold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from app.utils.helpers import get_column_types, split_features_target


def variance_reduction(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    X, y = split_features_target(df, target_col)
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    non_numeric = [c for c in X.columns if c not in numeric_cols]

    selector = VarianceThreshold(threshold=threshold)
    X_num = X[numeric_cols]
    selected = selector.fit_transform(X_num)
    selected_cols = [
        col for col, keep in zip(numeric_cols, selector.get_support()) if keep
    ]
    removed = [c for c in numeric_cols if c not in selected_cols]

    result_X = pd.DataFrame(selected, columns=selected_cols, index=X.index)
    if non_numeric:
        result_X = pd.concat([result_X, X[non_numeric]], axis=1)
    result = pd.concat([result_X, y], axis=1)

    return result, {
        "removed_low_variance": removed,
        "kept_columns": selected_cols,
        "threshold": threshold,
    }


def apply_pca(
    df: pd.DataFrame,
    target_col: str,
    n_components: int | None = None,
    variance_threshold: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    X, y = split_features_target(df, target_col)
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    X_num = X[numeric_cols].fillna(0)

    if n_components is None:
        pca_full = PCA()
        pca_full.fit(X_num)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)

    pca = PCA(n_components=min(n_components, len(numeric_cols)))
    transformed = pca.fit_transform(X_num)
    pca_cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
    result_X = pd.DataFrame(transformed, columns=pca_cols, index=X.index)

    non_numeric = [c for c in X.columns if c not in numeric_cols]
    if non_numeric:
        result_X = pd.concat([result_X, X[non_numeric].reset_index(drop=True)], axis=1)

    result = pd.concat([result_X, y.reset_index(drop=True)], axis=1)

    return result, {
        "n_components": transformed.shape[1],
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_variance_explained": round(sum(pca.explained_variance_ratio_) * 100, 2),
        "original_features": len(numeric_cols),
    }


def pca_scree_plot(variance_ratios: list[float]):
    import numpy as np

    cumvar = np.cumsum(variance_ratios)
    fig = px.bar(
        x=[f"PC{i+1}" for i in range(len(variance_ratios))],
        y=variance_ratios,
        title="PCA Explained Variance per Component",
        labels={"x": "Component", "y": "Variance Ratio"},
    )
    fig.add_scatter(
        x=[f"PC{i+1}" for i in range(len(cumvar))],
        y=cumvar,
        mode="lines+markers",
        name="Cumulative",
        yaxis="y2",
    )
    fig.update_layout(
        yaxis2=dict(title="Cumulative", overlaying="y", side="right"),
        height=400,
    )
    return fig
