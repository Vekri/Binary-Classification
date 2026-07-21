"""Business insights generation from model results."""

from __future__ import annotations

from typing import Any

import pandas as pd


def generate_business_insights(
    df: pd.DataFrame,
    target_col: str,
    training_result: dict[str, Any],
    explanations: dict[str, Any],
    quality_score: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = training_result["metrics"]
    model_name = training_result["model_name"]
    target_dist = df[target_col].value_counts(normalize=True)

    insights: list[dict[str, str]] = []

    # Class balance
    minority_pct = target_dist.min() * 100
    if minority_pct < 20:
        insights.append(
            {
                "category": "Data Balance",
                "insight": f"Target class is imbalanced ({minority_pct:.1f}% minority). "
                "Consider SMOTE or class-weighted models for production.",
                "priority": "High",
            }
        )
    else:
        insights.append(
            {
                "category": "Data Balance",
                "insight": f"Target classes are reasonably balanced (minority: {minority_pct:.1f}%).",
                "priority": "Low",
            }
        )

    # Model performance
    if metrics["roc_auc"] >= 0.85:
        perf = "excellent"
        priority = "Low"
    elif metrics["roc_auc"] >= 0.75:
        perf = "good"
        priority = "Medium"
    else:
        perf = "needs improvement"
        priority = "High"

    insights.append(
        {
            "category": "Model Performance",
            "insight": f"{model_name} achieves {perf} performance with "
            f"ROC-AUC of {metrics['roc_auc']:.2%}, accuracy of {metrics['accuracy']:.2%}, "
            f"and F1 score of {metrics['f1']:.2%}.",
            "priority": priority,
        }
    )

    # Precision vs Recall tradeoff
    if metrics["precision"] > metrics["recall"] + 0.1:
        insights.append(
            {
                "category": "Business Trade-off",
                "insight": "Model favors precision over recall — fewer false positives. "
                "Suitable when cost of false alarms is high.",
                "priority": "Medium",
            }
        )
    elif metrics["recall"] > metrics["precision"] + 0.1:
        insights.append(
            {
                "category": "Business Trade-off",
                "insight": "Model favors recall over precision — catches more positives. "
                "Suitable when missing a positive case is costly.",
                "priority": "Medium",
            }
        )

    # Top drivers
    if "feature_importance" in explanations:
        top_features = explanations["feature_importance"].head(3)["feature"].tolist()
        insights.append(
            {
                "category": "Key Drivers",
                "insight": f"Top predictive features: {', '.join(top_features)}. "
                "Focus data collection and monitoring on these variables.",
                "priority": "High",
            }
        )

    # Data quality
    if quality_score:
        grade = quality_score.get("grade", "N/A")
        score = quality_score.get("overall_score", 0)
        if score < 70:
            insights.append(
                {
                    "category": "Data Quality",
                    "insight": f"Data quality score is {score} (Grade {grade}). "
                    "Improve data collection processes before deploying to production.",
                    "priority": "High",
                }
            )
        else:
            insights.append(
                {
                    "category": "Data Quality",
                    "insight": f"Data quality score is {score} (Grade {grade}) — acceptable for modeling.",
                    "priority": "Low",
                }
            )

    # Deployment recommendation
    if metrics["roc_auc"] >= 0.75 and (quality_score is None or quality_score.get("overall_score", 0) >= 70):
        deploy = "Recommended for pilot deployment with monitoring."
    else:
        deploy = "Not recommended for deployment yet — iterate on data quality and model tuning."

    insights.append(
        {
            "category": "Deployment",
            "insight": deploy,
            "priority": "High",
        }
    )

    return {
        "insights": insights,
        "insights_df": pd.DataFrame(insights),
        "summary": {
            "model": model_name,
            "roc_auc": metrics["roc_auc"],
            "rows": len(df),
            "features": len(df.columns) - 1,
            "deployment_ready": metrics["roc_auc"] >= 0.75,
        },
    }
