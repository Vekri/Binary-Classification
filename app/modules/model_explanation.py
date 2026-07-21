"""Model explanation using SHAP and feature importance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve


def explain_model(
    training_result: dict[str, Any],
    max_shap_rows: int = 50,
) -> dict[str, Any]:
    model = training_result["model"]
    X_test = training_result["X_test"]
    feature_names = training_result["feature_names"]

    explanations: dict[str, Any] = {"feature_names": feature_names}

    # Always compute cheap native importance first
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        explanations["feature_importance"] = imp
        explanations["importance_chart"] = _importance_chart(imp)
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_).ravel()
        if len(coef) == len(feature_names):
            imp = pd.DataFrame(
                {"feature": feature_names, "importance": coef}
            ).sort_values("importance", ascending=False)
            explanations["feature_importance"] = imp
            explanations["importance_chart"] = _importance_chart(imp)

    # Fast SHAP: TreeExplainer / LinearExplainer only — skip slow KernelExplainer
    try:
        import shap

        sample = X_test.iloc[: min(max_shap_rows, len(X_test))]
        model_name = type(model).__name__

        if model_name in (
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "DecisionTreeClassifier",
            "AdaBoostClassifier",
        ):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
            if isinstance(shap_values, list):
                vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                vals = shap_values
                if getattr(vals, "ndim", 1) == 3:
                    vals = vals[:, :, 1]
        elif model_name == "LogisticRegression":
            explainer = shap.LinearExplainer(model, sample)
            shap_values = explainer.shap_values(sample)
            vals = shap_values
        else:
            raise RuntimeError(f"Fast SHAP not available for {model_name}")

        mean_abs = np.abs(vals).mean(axis=0).ravel()
        shap_df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        explanations["shap_importance"] = shap_df
        explanations["shap_chart"] = _shap_chart(shap_df)
    except Exception as exc:
        explanations["shap_error"] = str(exc)
        # Fall back: reuse feature importance as SHAP chart if available
        if "feature_importance" in explanations and "shap_chart" not in explanations:
            fb = explanations["feature_importance"].rename(
                columns={"importance": "mean_abs_shap"}
            )
            explanations["shap_importance"] = fb
            explanations["shap_chart"] = _shap_chart(fb)

    y_test = training_result["y_test"]
    y_prob = training_result["y_prob"]
    explanations["confusion_matrix"] = confusion_matrix(
        y_test, training_result["y_pred"]
    )
    explanations["confusion_chart"] = _confusion_chart(
        explanations["confusion_matrix"]
    )
    explanations["roc_chart"] = _roc_chart(y_test, y_prob)

    return explanations


def _importance_chart(imp_df: pd.DataFrame, top_n: int = 15):
    top = imp_df.head(top_n)
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance",
        color="importance",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=450, yaxis=dict(autorange="reversed"), showlegend=False)
    return fig


def _shap_chart(shap_df: pd.DataFrame, top_n: int = 15):
    top = shap_df.head(top_n)
    fig = px.bar(
        top,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="SHAP Feature Importance",
        color="mean_abs_shap",
        color_continuous_scale="Plasma",
    )
    fig.update_layout(height=450, yaxis=dict(autorange="reversed"), showlegend=False)
    return fig


def _confusion_chart(cm: np.ndarray):
    labels = ["Predicted 0", "Predicted 1"]
    actual = ["Actual 0", "Actual 1"]
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=actual,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
    )
    fig.update_layout(height=400)
    return fig


def _roc_chart(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
    )
    return fig
