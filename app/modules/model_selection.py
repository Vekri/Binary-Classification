"""Model selection and comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.utils.helpers import safe_encode_target, split_features_target, train_test_split_data

# Fast defaults for interactive use (skip SVM — very slow with probability=True)
FAST_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "Decision Tree",
    "Naive Bayes",
    "Gradient Boosting",
]

ALL_MODELS = FAST_MODELS + ["KNN", "AdaBoost", "SVM"]


def get_model_catalog(fast: bool = True) -> dict[str, Any]:
    catalog = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, solver="lbfgs", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=40 if fast else 100,
            max_depth=10 if fast else None,
            random_state=42,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=40 if fast else 100,
            max_depth=3,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(n_estimators=30 if fast else 50, random_state=42),
        "SVM": SVC(probability=True, kernel="linear", max_iter=1000, random_state=42),
    }
    return catalog


def _maybe_sample(X: pd.DataFrame, y: pd.Series, max_rows: int = 2000):
    if len(X) <= max_rows:
        return X, y
    idx = X.sample(n=max_rows, random_state=42).index
    return X.loc[idx], y.loc[idx]


def compare_models(
    df: pd.DataFrame,
    target_col: str,
    cv_folds: int = 3,
    models: list[str] | None = None,
    fast: bool = True,
    max_rows: int = 2000,
) -> tuple[dict[str, Any], pd.DataFrame]:
    X, y = split_features_target(df, target_col)
    y_enc, label_map = safe_encode_target(y)
    X_proc = pd.get_dummies(X, drop_first=True).fillna(0)
    X_proc, y_enc = _maybe_sample(X_proc, y_enc, max_rows=max_rows)

    catalog = get_model_catalog(fast=fast)
    selected = models or (FAST_MODELS if fast else ALL_MODELS)
    results = []

    for name in selected:
        if name not in catalog:
            continue
        model = catalog[name]
        try:
            # n_jobs=1 avoids CPU thrashing from nested parallelism
            scores = cross_val_score(
                model, X_proc, y_enc, cv=cv_folds, scoring="roc_auc", n_jobs=1
            )
            results.append(
                {
                    "model": name,
                    "roc_auc_mean": round(float(np.nanmean(scores)), 4),
                    "roc_auc_std": round(float(np.nanstd(scores)), 4),
                    "status": "success",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "model": name,
                    "roc_auc_mean": 0,
                    "roc_auc_std": 0,
                    "status": f"failed: {exc}",
                }
            )

    results_df = pd.DataFrame(results)
    results_df["roc_auc_mean"] = pd.to_numeric(
        results_df["roc_auc_mean"], errors="coerce"
    ).fillna(0)
    results_df = results_df.sort_values("roc_auc_mean", ascending=False)
    best = results_df.iloc[0] if len(results_df) else None

    return {
        "best_model": best["model"] if best is not None else None,
        "label_map": label_map,
        "cv_folds": cv_folds,
    }, results_df


def train_best_model(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    test_size: float = 0.2,
    fast: bool = True,
) -> dict[str, Any]:
    X, y = split_features_target(df, target_col)
    y_enc, label_map = safe_encode_target(y)
    X_proc = pd.get_dummies(X, drop_first=True).fillna(0)
    feature_names = X_proc.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split_data(
        X_proc, y_enc, test_size=test_size
    )

    catalog = get_model_catalog(fast=fast)
    model = catalog[model_name]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4)
        if len(np.unique(y_test)) > 1
        else 0,
    }

    return {
        "model": model,
        "model_name": model_name,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "feature_names": feature_names,
        "label_map": label_map,
        "X_train": X_train,
        "y_train": y_train,
    }


def model_comparison_chart(results_df: pd.DataFrame):
    fig = px.bar(
        results_df,
        x="model",
        y="roc_auc_mean",
        error_y="roc_auc_std",
        title="Model Comparison (Cross-Validated ROC-AUC)",
        labels={"roc_auc_mean": "ROC-AUC"},
    )
    fig.update_layout(height=450, xaxis_tickangle=-30)
    return fig
