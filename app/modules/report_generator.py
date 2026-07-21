"""Executive PDF report generation."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from fpdf import FPDF


class ExecutiveReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Binary Classification - Executive Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Generated {datetime.now():%Y-%m-%d %H:%M}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(41, 98, 255)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f"  {title}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        # Helvetica is latin-1 only — strip unsupported characters
        safe = (
            text.replace("\u2014", "-")
            .replace("\u2013", "-")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2022", "-")
            .encode("latin-1", errors="replace")
            .decode("latin-1")
        )
        self.multi_cell(0, 6, safe)
        self.ln(2)


def generate_executive_report(
    pipeline_state: dict[str, Any],
    output_dir: str = "reports",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"executive_report_{timestamp}.pdf")

    pdf = ExecutiveReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Executive Summary
    pdf.section_title("Executive Summary")
    summary = pipeline_state.get("business_insights", {}).get("summary", {})
    metrics = pipeline_state.get("training_result", {}).get("metrics", {})
    quality = pipeline_state.get("quality_score", {})

    pdf.body_text(
        f"This report summarizes the binary classification analysis performed on "
        f"{summary.get('rows', 'N/A')} records with {summary.get('features', 'N/A')} features. "
        f"The best performing model is {summary.get('model', 'N/A')} with a ROC-AUC of "
        f"{metrics.get('roc_auc', 0):.2%}."
    )

    deploy = "YES" if summary.get("deployment_ready") else "NO"
    pdf.body_text(f"Deployment Recommendation: {deploy}")

    # Data Quality
    pdf.section_title("Data Quality Assessment")
    if quality:
        pdf.body_text(
            f"Overall Quality Score: {quality.get('overall_score', 'N/A')} (Grade: {quality.get('grade', 'N/A')})\n"
            f"  - Completeness: {quality.get('dimensions', {}).get('completeness', 'N/A')}%\n"
            f"  - Uniqueness: {quality.get('dimensions', {}).get('uniqueness', 'N/A')}%\n"
            f"  - Validity: {quality.get('dimensions', {}).get('validity', 'N/A')}%\n"
            f"  - Consistency: {quality.get('dimensions', {}).get('consistency', 'N/A')}%"
        )
    else:
        pdf.body_text("Data quality assessment was not performed.")

    # Model Performance
    pdf.section_title("Model Performance")
    if metrics:
        pdf.body_text(
            f"Accuracy:  {metrics.get('accuracy', 0):.2%}\n"
            f"Precision: {metrics.get('precision', 0):.2%}\n"
            f"Recall:    {metrics.get('recall', 0):.2%}\n"
            f"F1 Score:  {metrics.get('f1', 0):.2%}\n"
            f"ROC-AUC:   {metrics.get('roc_auc', 0):.2%}"
        )

    # Pipeline Steps
    pdf.section_title("Pipeline Steps Completed")
    steps = pipeline_state.get("completed_steps", [])
    for step in steps:
        pdf.body_text(f"  [done] {step}")

    # Business Insights
    pdf.section_title("Key Business Insights")
    insights = pipeline_state.get("business_insights", {}).get("insights", [])
    for item in insights:
        pdf.body_text(f"[{item['priority']}] {item['category']}: {item['insight']}")

    # Hyperparameter Tuning
    tuning = pipeline_state.get("tuning_result")
    if tuning and "best_params" in tuning:
        pdf.section_title("Hyperparameter Tuning")
        pdf.body_text(f"Model: {tuning.get('model_name', 'N/A')}")
        pdf.body_text(f"Best CV Score: {tuning.get('best_score', 'N/A')}")
        params = ", ".join(f"{k}={v}" for k, v in tuning.get("best_params", {}).items())
        pdf.body_text(f"Best Parameters: {params}")

    # Top Features
    explanations = pipeline_state.get("explanations", {})
    if "feature_importance" in explanations:
        pdf.section_title("Top Predictive Features")
        top = explanations["feature_importance"].head(10)
        for _, row in top.iterrows():
            pdf.body_text(f"  - {row['feature']}: {row['importance']:.4f}")

    pdf.output(filepath)
    return filepath
