"""Variable reduction via variance, correlation, VIF, PCA, and voting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression

from app.utils.helpers import split_features_target


def _numeric_matrix(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    X, y = split_features_target(df, target_col)
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    X_num = X[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return X_num, y, [c for c in X.columns if c not in numeric_cols]


def variance_reduction(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    X, y = split_features_target(df, target_col)
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    non_numeric = [c for c in X.columns if c not in numeric_cols]

    selector = VarianceThreshold(threshold=threshold)
    X_num = X[numeric_cols].fillna(0)
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
        "method": "Variance threshold",
        "removed_low_variance": removed,
        "kept_columns": selected_cols,
        "threshold": threshold,
    }


def correlation_reduction(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = 0.90,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop one feature from each highly correlated pair (keep first)."""
    X_num, y, non_numeric = _numeric_matrix(df, target_col)
    if X_num.shape[1] < 2:
        result = pd.concat([X_num, y], axis=1)
        return result, {
            "method": "Correlation",
            "removed": [],
            "kept_columns": X_num.columns.tolist(),
            "threshold": threshold,
            "pairs": [],
        }

    corr = X_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop: list[str] = []
    pairs: list[dict[str, Any]] = []
    for col in upper.columns:
        high = upper.index[upper[col] > threshold].tolist()
        for other in high:
            if other not in to_drop and col not in to_drop:
                to_drop.append(col)
                pairs.append(
                    {
                        "drop": col,
                        "keep": other,
                        "correlation": round(float(corr.loc[other, col]), 4),
                    }
                )

    kept = [c for c in X_num.columns if c not in to_drop]
    result_X = X_num[kept].copy()
    X_full, _ = split_features_target(df, target_col)
    if non_numeric:
        result_X = pd.concat([result_X, X_full[non_numeric]], axis=1)
    result = pd.concat([result_X, y], axis=1)

    return result, {
        "method": "Correlation",
        "removed": to_drop,
        "kept_columns": kept,
        "threshold": threshold,
        "pairs": pairs,
    }


def compute_vif_table(X_num: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF for each numeric column (no statsmodels dependency)."""
    cols = X_num.columns.tolist()
    if len(cols) < 2:
        return pd.DataFrame({"feature": cols, "vif": [1.0] * len(cols)})

    X = X_num.astype(float)
    rows = []
    for col in cols:
        y = X[col]
        X_i = X.drop(columns=[col])
        # Constant / near-constant columns → infinite VIF signal
        if y.std() < 1e-12:
            rows.append({"feature": col, "vif": float("inf")})
            continue
        try:
            r2 = float(LinearRegression().fit(X_i, y).score(X_i, y))
            r2 = min(max(r2, 0.0), 0.999999)
            vif = 1.0 / (1.0 - r2)
        except Exception:
            vif = float("inf")
        rows.append({"feature": col, "vif": round(vif, 4)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def vif_reduction(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = 10.0,
    max_iter: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Iteratively drop the highest-VIF feature until all VIFs are below threshold."""
    X_num, y, non_numeric = _numeric_matrix(df, target_col)
    removed: list[dict[str, Any]] = []
    remaining = X_num.copy()

    for _ in range(max_iter):
        if remaining.shape[1] < 2:
            break
        vif_df = compute_vif_table(remaining)
        top = vif_df.iloc[0]
        if float(top["vif"]) <= threshold:
            break
        drop_col = str(top["feature"])
        removed.append({"feature": drop_col, "vif": float(top["vif"])})
        remaining = remaining.drop(columns=[drop_col])

    kept = remaining.columns.tolist()
    result_X = remaining.copy()
    X_full, _ = split_features_target(df, target_col)
    if non_numeric:
        result_X = pd.concat([result_X, X_full[non_numeric]], axis=1)
    result = pd.concat([result_X, y], axis=1)

    return result, {
        "method": "VIF",
        "removed": removed,
        "kept_columns": kept,
        "threshold": threshold,
        "vif_table": compute_vif_table(remaining) if remaining.shape[1] else pd.DataFrame(),
    }


def voting_variable_reduction(
    df: pd.DataFrame,
    target_col: str,
    variance_threshold: float = 0.01,
    corr_threshold: float = 0.90,
    vif_threshold: float = 10.0,
    min_drop_votes: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Reduce variables by majority vote across methods:
    - Low variance
    - High pairwise correlation
    - High VIF (multicollinearity)

    A numeric feature is dropped when drop-votes >= min_drop_votes.
    Non-numeric columns are always retained.
    """
    X, y = split_features_target(df, target_col)
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    non_numeric = [c for c in X.columns if c not in numeric_cols]
    X_num = X[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    votes = {c: {"variance": 0, "correlation": 0, "vif": 0} for c in numeric_cols}

    # --- Variance votes ---
    if numeric_cols:
        variances = X_num.var()
        for col in numeric_cols:
            if float(variances.get(col, 0)) < variance_threshold:
                votes[col]["variance"] = 1

    # --- Correlation votes (drop first of each high-corr pair) ---
    corr_pairs: list[dict[str, Any]] = []
    if len(numeric_cols) >= 2:
        corr = X_num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        marked: set[str] = set()
        for col in upper.columns:
            high = upper.index[upper[col] > corr_threshold].tolist()
            for other in high:
                if col in marked or other in marked:
                    continue
                votes[col]["correlation"] = 1
                marked.add(col)
                corr_pairs.append(
                    {
                        "drop": col,
                        "keep": other,
                        "correlation": round(float(corr.loc[other, col]), 4),
                    }
                )

    # --- VIF votes (iterative high-VIF flags, without mutating final set yet) ---
    vif_flags: list[dict[str, Any]] = []
    remaining = X_num.copy()
    for _ in range(min(50, max(0, len(numeric_cols) - 1))):
        if remaining.shape[1] < 2:
            break
        vif_df = compute_vif_table(remaining)
        top = vif_df.iloc[0]
        if float(top["vif"]) <= vif_threshold:
            break
        drop_col = str(top["feature"])
        votes[drop_col]["vif"] = 1
        vif_flags.append({"feature": drop_col, "vif": float(top["vif"])})
        remaining = remaining.drop(columns=[drop_col])

    vote_rows = []
    to_drop = []
    for col, v in votes.items():
        total = int(v["variance"] + v["correlation"] + v["vif"])
        drop = total >= min_drop_votes
        if drop:
            to_drop.append(col)
        vote_rows.append(
            {
                "feature": col,
                "variance_vote": v["variance"],
                "correlation_vote": v["correlation"],
                "vif_vote": v["vif"],
                "drop_votes": total,
                "decision": "DROP" if drop else "KEEP",
            }
        )

    vote_table = pd.DataFrame(vote_rows).sort_values(
        ["drop_votes", "feature"], ascending=[False, True]
    )
    kept = [c for c in numeric_cols if c not in to_drop]
    result_X = X_num[kept].copy()
    if non_numeric:
        result_X = pd.concat([result_X, X[non_numeric]], axis=1)
    result = pd.concat([result_X, y], axis=1)

    return result, {
        "method": "Voting (variance + correlation + VIF)",
        "removed": to_drop,
        "kept_columns": kept + non_numeric,
        "min_drop_votes": min_drop_votes,
        "thresholds": {
            "variance": variance_threshold,
            "correlation": corr_threshold,
            "vif": vif_threshold,
        },
        "vote_table": vote_table,
        "correlation_pairs": corr_pairs,
        "vif_flags": vif_flags,
        "n_methods": 3,
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
        "method": "PCA",
        "n_components": transformed.shape[1],
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_variance_explained": round(sum(pca.explained_variance_ratio_) * 100, 2),
        "original_features": len(numeric_cols),
    }


def pca_scree_plot(variance_ratios: list[float]):
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


def vote_summary_chart(vote_table: pd.DataFrame):
    if vote_table is None or vote_table.empty:
        return None
    plot_df = vote_table.sort_values("drop_votes", ascending=True).tail(25)
    fig = px.bar(
        plot_df,
        x="drop_votes",
        y="feature",
        color="decision",
        orientation="h",
        title="Variable Reduction Votes (top features)",
        color_discrete_map={"DROP": "#e07a5f", "KEEP": "#3dcfb6"},
        labels={"drop_votes": "Drop votes (of 3)", "feature": "Feature"},
    )
    fig.update_layout(height=max(350, len(plot_df) * 22))
    return fig
