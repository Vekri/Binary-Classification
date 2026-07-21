"""Feature selection methods."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)

from app.utils.helpers import safe_encode_target, split_features_target


def select_features_univariate(
    df: pd.DataFrame,
    target_col: str,
    k: int = 10,
    score_func: str = "f_classif",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    X, y = split_features_target(df, target_col)
    y_enc, _ = safe_encode_target(y)
    X_num = pd.get_dummies(X, drop_first=True)
    X_num = X_num.fillna(0)

    func = f_classif if score_func == "f_classif" else mutual_info_classif
    k = min(k, X_num.shape[1])
    selector = SelectKBest(score_func=func, k=k)
    selector.fit(X_num, y_enc)
    scores = selector.scores_
    selected_mask = selector.get_support()
    selected_cols = X_num.columns[selected_mask].tolist()

    result = pd.concat([X_num[selected_cols], y], axis=1)
    ranking = pd.DataFrame(
        {"feature": X_num.columns, "score": scores, "selected": selected_mask}
    ).sort_values("score", ascending=False)

    return result, {
        "method": f"SelectKBest ({score_func})",
        "k": k,
        "selected_features": selected_cols,
        "ranking": ranking,
    }


def select_features_rfe(
    df: pd.DataFrame,
    target_col: str,
    n_features: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    X, y = split_features_target(df, target_col)
    y_enc, _ = safe_encode_target(y)
    X_num = pd.get_dummies(X, drop_first=True).fillna(0)
    n_features = min(n_features, X_num.shape[1])

    estimator = RandomForestClassifier(
        n_estimators=25, max_depth=8, random_state=42, n_jobs=1
    )
    # step=0.2 removes 20% of features each iteration (much faster than step=1)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=0.2)
    rfe.fit(X_num, y_enc)
    selected_cols = X_num.columns[rfe.support_].tolist()
    result = pd.concat([X_num[selected_cols], y], axis=1)

    ranking = pd.DataFrame(
        {"feature": X_num.columns, "rank": rfe.ranking_, "selected": rfe.support_}
    ).sort_values("rank")

    return result, {
        "method": "RFE (RandomForest)",
        "n_features": n_features,
        "selected_features": selected_cols,
        "ranking": ranking,
    }


def select_features_importance(
    df: pd.DataFrame,
    target_col: str,
    threshold: str = "median",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    X, y = split_features_target(df, target_col)
    y_enc, _ = safe_encode_target(y)
    X_num = pd.get_dummies(X, drop_first=True).fillna(0)

    rf = RandomForestClassifier(
        n_estimators=40, max_depth=10, random_state=42, n_jobs=1
    )
    rf.fit(X_num, y_enc)
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    selected_cols = X_num.columns[selector.get_support()].tolist()
    result = pd.concat([X_num[selected_cols], y], axis=1)

    importance = pd.DataFrame(
        {"feature": X_num.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    return result, {
        "method": "SelectFromModel (RF importance)",
        "selected_features": selected_cols,
        "importance": importance,
    }


def feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 15):
    top = importance_df.head(top_n)
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Feature Importances",
    )
    fig.update_layout(height=450, yaxis=dict(autorange="reversed"))
    return fig
