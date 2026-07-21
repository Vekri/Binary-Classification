"""Binary Classification ML Pipeline — Streamlit Web App."""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.modules.business_insights import generate_business_insights
from app.modules.data_profiling import generate_profile_report, run_profiling
from app.modules.data_quality import column_quality_report, compute_quality_score, quality_radar_chart
from app.modules.duplicates import duplicate_chart, duplicate_summary, find_duplicates, remove_duplicates
from app.modules.encoding import apply_encoding, recommend_encoding
from app.modules.feature_engineering import engineer_features
from app.modules.feature_selection import (
    feature_importance_chart,
    select_features_importance,
    select_features_rfe,
    select_features_univariate,
)
from app.modules.hyperparameter_tuning import run_hyperparameter_tuning
from app.modules.missing_values import (
    apply_missing_treatment,
    missing_bar_chart,
    missing_value_summary,
    recommend_missing_strategy,
)
from app.modules.model_explanation import explain_model
from app.modules.model_selection import (
    compare_models,
    model_comparison_chart,
    train_best_model,
)
from app.modules.outliers import outlier_boxplots, outlier_summary, treat_outliers
from app.modules.report_generator import generate_executive_report
from app.modules.scaling import apply_scaling, recommend_scaling
from app.modules.variable_reduction import apply_pca, pca_scree_plot, variance_reduction

st.set_page_config(
    page_title="Binary Classification ML Pipeline",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 700; color: #2962FF; }
    .step-done { color: #4CAF50; font-weight: bold; }
    div[data-testid="stSidebar"] { background-color: #f8f9fa; }
    section[data-testid="stSidebar"] label p {
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.4rem 0.3rem !important;
        margin-bottom: 0.2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

STEPS = [
    "1. Profiling",
    "2. Data Quality",
    "3. Missing Values",
    "4. Outliers",
    "5. Duplicates",
    "6. Encoding",
    "7. Scaling",
    "8. Feature Engineering",
    "9. Variable Reduction",
    "10. Feature Selection",
    "11. Model Selection",
    "12. Hyperparameter Tuning",
    "13. Model Explanation",
    "14. Business Insights",
    "15. Executive Report",
]

DEFAULTS = {
    "raw_df": None,
    "processed_df": None,
    "target_col": None,
    "quality_score": None,
    "model_results": None,
    "training_result": None,
    "explanations": None,
    "business_insights": None,
    "tuning_result": None,
    "completed_steps": [],
    "pipeline_log": [],
    "nav_step": STEPS[0],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def log_step(step: str):
    if step not in st.session_state.completed_steps:
        st.session_state.completed_steps.append(step)


def get_active_df() -> pd.DataFrame | None:
    return st.session_state.processed_df or st.session_state.raw_df


with st.sidebar:
    st.markdown("## 🎯 ML Pipeline")
    st.markdown("---")

    fast_mode = st.toggle(
        "⚡ Fast mode",
        value=True,
        help="Fewer models, lighter trees, quicker SHAP. Turn off for full accuracy.",
    )
    st.session_state.fast_mode = fast_mode

    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded:
        try:
            loaded = pd.read_csv(uploaded)
            st.session_state.raw_df = loaded
            st.session_state.processed_df = None
            st.success(f"Loaded {len(loaded):,} rows × {len(loaded.columns)} cols")
        except Exception as exc:
            st.error(f"Failed to load: {exc}")

    if st.session_state.raw_df is not None:
        raw = st.session_state.raw_df
        st.session_state.target_col = st.selectbox(
            "Target column (binary)",
            options=raw.columns.tolist(),
            index=len(raw.columns) - 1,
        )

        if st.button("Reset pipeline", type="secondary"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v if k != "raw_df" else st.session_state.raw_df
            st.rerun()

        st.markdown("---")
        st.markdown("### Pipeline steps")
        st.caption("Click a step — no horizontal dragging")
        current = st.session_state.get("nav_step", STEPS[0])
        if current not in STEPS:
            current = STEPS[0]
        st.session_state.nav_step = st.radio(
            "Go to step",
            STEPS,
            index=STEPS.index(current),
            label_visibility="collapsed",
        )

    st.markdown("---")
    st.markdown("**Completed steps:**")
    for done in st.session_state.completed_steps:
        st.markdown(f'<span class="step-done">✓ {done}</span>', unsafe_allow_html=True)
    if not st.session_state.completed_steps:
        st.caption("No steps completed yet")

st.markdown('<p class="main-header">Binary Classification ML Pipeline</p>', unsafe_allow_html=True)
st.caption(
    "End-to-end data science workflow: profiling → preprocessing → modeling → insights → executive report"
)

if st.session_state.raw_df is None:
    st.info("👈 Upload a CSV file in the sidebar to begin.")
    st.markdown(
        """
        ### Features
        | Category | Capabilities |
        |----------|-------------|
        | **Data Understanding** | Profiling, quality scoring, duplicate detection |
        | **Preprocessing** | Missing values, outliers, encoding, scaling |
        | **Feature Engineering** | Auto features, variable reduction, feature selection |
        | **Modeling** | Model comparison, hyperparameter tuning (Optuna) |
        | **Insights** | SHAP explanations, business insights, PDF executive report |
        """
    )
    st.stop()

df = get_active_df()
target = st.session_state.target_col
step = st.session_state.get("nav_step", STEPS[0])
step_idx = STEPS.index(step) if step in STEPS else 0

nav_l, nav_c, nav_r = st.columns([1, 3, 1])
with nav_l:
    if st.button("← Previous", disabled=step_idx == 0, use_container_width=True):
        st.session_state.nav_step = STEPS[step_idx - 1]
        st.rerun()
with nav_c:
    st.markdown(
        f"<div style='text-align:center;font-size:1.2rem;font-weight:600;"
        f"padding:0.45rem 0'>{step}</div>",
        unsafe_allow_html=True,
    )
with nav_r:
    if st.button("Next →", disabled=step_idx >= len(STEPS) - 1, use_container_width=True):
        st.session_state.nav_step = STEPS[step_idx + 1]
        st.rerun()

st.progress((step_idx + 1) / len(STEPS), text=f"Step {step_idx + 1} of {len(STEPS)}")
st.markdown("---")

if step == STEPS[0]:
    st.subheader("Data Profiling")
    if st.button("Run profiling", key="btn_profile"):
        with st.spinner("Profiling data..."):
            result = run_profiling(df)
            st.session_state.profile_result = result
            log_step("Data Profiling")

    if "profile_result" in st.session_state:
        r = st.session_state.profile_result
        s = r["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{s['rows']:,}")
        c2.metric("Columns", s["columns"])
        c3.metric("Missing %", f"{s['missing_pct']}%")
        c4.metric("Duplicates", s["duplicate_rows"])
        st.plotly_chart(r["missing_heatmap"], use_container_width=True)
        st.plotly_chart(r["correlation"], use_container_width=True)
        st.dataframe(r["stats"], use_container_width=True)
        if st.button("Generate full HTML profile (ydata-profiling)", key="btn_html"):
            with st.spinner("Generating report..."):
                report = generate_profile_report(st.session_state.raw_df, minimal=True)
                html = report.to_html()
                st.download_button(
                    "Download HTML Profile",
                    html,
                    file_name="data_profile.html",
                    mime="text/html",
                )

elif step == STEPS[1]:
    st.subheader("Data Quality Scoring")
    if st.button("Compute quality score", key="btn_quality"):
        st.session_state.quality_score = compute_quality_score(df)
        log_step("Data Quality Scoring")
    if st.session_state.quality_score:
        qs = st.session_state.quality_score
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Overall Score", f"{qs['overall_score']}/100")
            st.metric("Grade", qs["grade"])
            for dim, val in qs["dimensions"].items():
                st.progress(val / 100, text=f"{dim.title()}: {val}%")
        with c2:
            st.plotly_chart(quality_radar_chart(qs), use_container_width=True)
        st.dataframe(column_quality_report(df), use_container_width=True)

elif step == STEPS[2]:
    st.subheader("Missing Value Treatment")
    st.dataframe(missing_value_summary(df), use_container_width=True)
    chart = missing_bar_chart(df)
    if chart:
        st.plotly_chart(chart, use_container_width=True)
    st.markdown("**Recommendations**")
    st.dataframe(recommend_missing_strategy(df), use_container_width=True)
    col1, col2 = st.columns(2)
    use_knn = col1.checkbox("Use KNN imputation for numeric columns")
    if col2.button("Apply recommended treatment", key="btn_missing"):
        with st.spinner("Treating missing values..."):
            treated, log = apply_missing_treatment(df, use_knn=use_knn)
            st.session_state.processed_df = treated
            st.success(f"Done. Remaining missing: {log['remaining_missing']}")
            for action in log["actions"]:
                st.write(f"• {action}")
            log_step("Missing Value Treatment")
            st.rerun()

elif step == STEPS[3]:
    st.subheader("Outlier Detection")
    method = st.selectbox("Detection method", ["iqr", "zscore"])
    threshold = st.slider("Threshold", 1.0, 4.0, 1.5, 0.1)
    exclude = [target] if target else []
    st.dataframe(
        outlier_summary(df, method=method, threshold=threshold, exclude_cols=exclude),
        use_container_width=True,
    )
    st.plotly_chart(outlier_boxplots(df), use_container_width=True)
    treat_method = st.selectbox("Treatment", ["cap", "remove"])
    if st.button("Apply outlier treatment", key="btn_outliers"):
        treated, log = treat_outliers(
            df,
            method=treat_method,
            detection=method,
            threshold=threshold,
            exclude_cols=exclude,
        )
        st.session_state.processed_df = treated
        st.success(f"Treated {log['outliers_treated']} outlier values")
        log_step("Outlier Detection & Treatment")
        st.rerun()

elif step == STEPS[4]:
    st.subheader("Duplicate Detection")
    summary = duplicate_summary(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicate rows", summary["total_duplicate_rows"])
    c2.metric("Duplicate %", f"{summary['duplicate_pct']}%")
    c3.metric("Unique rows", summary["unique_rows"])
    st.plotly_chart(duplicate_chart(df), use_container_width=True)
    if summary["total_duplicate_rows"] > 0:
        st.dataframe(find_duplicates(df).head(50), use_container_width=True)
        if st.button("Remove duplicates", key="btn_dup"):
            treated, log = remove_duplicates(df)
            st.session_state.processed_df = treated
            st.success(f"Removed {log['rows_removed']} duplicate rows")
            log_step("Duplicate Detection")
            st.rerun()

elif step == STEPS[5]:
    st.subheader("Encoding Recommendations")
    st.dataframe(recommend_encoding(df, target), use_container_width=True)
    if st.button("Apply recommended encoding", key="btn_encode"):
        with st.spinner("Encoding categorical features..."):
            treated, log = apply_encoding(df, target)
            st.session_state.processed_df = treated
            st.success(f"Encoded {len(log['columns_encoded'])} columns")
            log_step("Encoding")
            st.rerun()

elif step == STEPS[6]:
    st.subheader("Scaling Recommendations")
    st.dataframe(recommend_scaling(df), use_container_width=True)
    global_scaler = st.selectbox(
        "Global scaler (overrides per-column)",
        ["none", "standard", "minmax", "robust", "maxabs"],
    )
    if st.button("Apply scaling", key="btn_scale"):
        treated, log = apply_scaling(df, target, global_scaler=global_scaler)
        st.session_state.processed_df = treated
        st.success("Scaling applied")
        log_step("Scaling")
        st.rerun()

elif step == STEPS[7]:
    st.subheader("Feature Engineering")
    c1, c2, c3, c4, c5 = st.columns(5)
    opts = {
        "interactions": c1.checkbox("Interactions", True),
        "polynomial": c2.checkbox("Polynomial", True),
        "log_transform": c3.checkbox("Log transform", True),
        "binning": c4.checkbox("Binning", True),
        "datetime_features": c5.checkbox("Datetime", True),
    }
    if st.button("Engineer features", key="btn_fe"):
        with st.spinner("Engineering features..."):
            treated, log = engineer_features(df, target, opts)
            st.session_state.processed_df = treated
            st.success(f"Created {len(log['new_features'])} new features")
            if log["new_features"]:
                st.write(", ".join(log["new_features"][:20]))
            log_step("Feature Engineering")
            st.rerun()

elif step == STEPS[8]:
    st.subheader("Variable Reduction")
    method = st.radio("Method", ["Variance threshold", "PCA"])
    if method == "Variance threshold":
        thresh = st.slider("Variance threshold", 0.0, 0.1, 0.01, 0.005)
        if st.button("Apply variance reduction", key="btn_var"):
            treated, log = variance_reduction(df, target, threshold=thresh)
            st.session_state.processed_df = treated
            st.success(f"Removed {len(log['removed_low_variance'])} low-variance columns")
            log_step("Variable Reduction")
            st.rerun()
    else:
        var_thresh = st.slider("Variance to retain", 0.5, 0.99, 0.95, 0.01)
        if st.button("Apply PCA", key="btn_pca"):
            treated, log = apply_pca(df, target, variance_threshold=var_thresh)
            st.session_state.processed_df = treated
            st.success(
                f"Reduced to {log['n_components']} components "
                f"({log['total_variance_explained']}% variance)"
            )
            st.plotly_chart(
                pca_scree_plot(log["explained_variance_ratio"]),
                use_container_width=True,
            )
            log_step("Variable Reduction (PCA)")
            st.rerun()

elif step == STEPS[9]:
    st.subheader("Feature Selection")
    fs_method = st.selectbox("Method", ["Univariate (SelectKBest)", "RFE", "Importance-based"])
    k = st.slider("Number of features", 3, 50, 10)
    if st.button("Run feature selection", key="btn_fs"):
        with st.spinner("Selecting features..."):
            if fs_method.startswith("Univariate"):
                treated, log = select_features_univariate(df, target, k=k)
            elif fs_method == "RFE":
                treated, log = select_features_rfe(df, target, n_features=k)
            else:
                treated, log = select_features_importance(df, target)
            st.session_state.processed_df = treated
            st.session_state.fs_log = log
            st.success(
                f"Selected {len(log['selected_features'])} features via {log['method']}"
            )
            log_step("Feature Selection")
            st.rerun()
    if "fs_log" in st.session_state:
        log = st.session_state.fs_log
        if "ranking" in log:
            st.dataframe(log["ranking"].head(20), use_container_width=True)
        if "importance" in log:
            st.plotly_chart(
                feature_importance_chart(log["importance"]),
                use_container_width=True,
            )

elif step == STEPS[10]:
    st.subheader("Model Selection")
    fast = st.session_state.get("fast_mode", True)
    if fast:
        st.caption("Fast mode: comparing 5 quick models with 3-fold CV")
    cv_folds = st.slider("CV folds", 2, 10, 3 if fast else 5)
    if st.button("Compare models", key="btn_models"):
        with st.spinner("Cross-validating models..."):
            meta, results_df = compare_models(df, target, cv_folds=cv_folds, fast=fast)
            st.session_state.model_results = results_df
            st.session_state.best_model_name = meta["best_model"]
            log_step("Model Selection")
            st.success(f"Best model: **{meta['best_model']}**")
    if st.session_state.model_results is not None:
        st.plotly_chart(
            model_comparison_chart(st.session_state.model_results),
            use_container_width=True,
        )
        st.dataframe(st.session_state.model_results, use_container_width=True)
        model_to_train = st.selectbox(
            "Model to train",
            st.session_state.model_results["model"].tolist(),
            index=0,
        )
        if st.button("Train selected model", key="btn_train"):
            with st.spinner(f"Training {model_to_train}..."):
                st.session_state.training_result = train_best_model(
                    df, target, model_to_train, fast=fast
                )
                log_step(f"Trained: {model_to_train}")
                m = st.session_state.training_result["metrics"]
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{m['accuracy']:.2%}")
                c2.metric("Precision", f"{m['precision']:.2%}")
                c3.metric("Recall", f"{m['recall']:.2%}")
                c4.metric("F1", f"{m['f1']:.2%}")
                c5.metric("ROC-AUC", f"{m['roc_auc']:.2%}")

elif step == STEPS[11]:
    st.subheader("Hyperparameter Tuning (Optuna)")
    tune_model = st.selectbox(
        "Model to tune",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"],
    )
    n_trials = st.slider(
        "Number of trials",
        5,
        50,
        8 if st.session_state.get("fast_mode", True) else 20,
    )
    st.caption("Tip: Logistic Regression tunes fastest.")
    if st.button("Run tuning", key="btn_tune"):
        with st.spinner(f"Tuning {tune_model} ({n_trials} trials)..."):
            result = run_hyperparameter_tuning(df, target, tune_model, n_trials=n_trials)
            st.session_state.tuning_result = result
            log_step("Hyperparameter Tuning")
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Best CV ROC-AUC: {result['best_score']}")
                st.json(result["best_params"])

elif step == STEPS[12]:
    st.subheader("Model Explanation (SHAP)")
    if st.session_state.training_result is None:
        st.warning("Train a model first in Model Selection.")
    else:
        if st.button("Generate explanations", key="btn_explain"):
            with st.spinner("Computing SHAP values..."):
                st.session_state.explanations = explain_model(
                    st.session_state.training_result
                )
                log_step("Model Explanation")
        if st.session_state.explanations:
            exp = st.session_state.explanations
            col1, col2 = st.columns(2)
            with col1:
                if "importance_chart" in exp:
                    st.plotly_chart(exp["importance_chart"], use_container_width=True)
            with col2:
                if "shap_chart" in exp:
                    st.plotly_chart(exp["shap_chart"], use_container_width=True)
                elif "shap_error" in exp:
                    st.info(f"SHAP unavailable: {exp['shap_error']}")
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(exp["confusion_chart"], use_container_width=True)
            with col4:
                st.plotly_chart(exp["roc_chart"], use_container_width=True)

elif step == STEPS[13]:
    st.subheader("Business Insights")
    if st.session_state.training_result is None:
        st.warning("Train a model first.")
    else:
        if st.button("Generate insights", key="btn_insights"):
            if st.session_state.explanations is None:
                st.session_state.explanations = explain_model(
                    st.session_state.training_result
                )
            st.session_state.business_insights = generate_business_insights(
                df,
                target,
                st.session_state.training_result,
                st.session_state.explanations,
                st.session_state.quality_score,
            )
            log_step("Business Insights")
        if st.session_state.business_insights:
            bi = st.session_state.business_insights
            s = bi["summary"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model", s["model"])
            c2.metric("ROC-AUC", f"{s['roc_auc']:.2%}")
            c3.metric("Records", f"{s['rows']:,}")
            c4.metric("Deployment", "Ready" if s["deployment_ready"] else "Not Ready")
            for item in bi["insights"]:
                icon = (
                    "🔴"
                    if item["priority"] == "High"
                    else "🟡"
                    if item["priority"] == "Medium"
                    else "🟢"
                )
                st.markdown(
                    f"{icon} **{item['category']}** ({item['priority']}): {item['insight']}"
                )

elif step == STEPS[14]:
    st.subheader("Executive Report Generation")
    st.markdown("Generate a PDF executive summary of the entire pipeline.")
    if st.button("Generate PDF Report", key="btn_report", type="primary"):
        pipeline_state = {
            "quality_score": st.session_state.quality_score,
            "training_result": st.session_state.training_result,
            "explanations": st.session_state.explanations,
            "business_insights": st.session_state.business_insights,
            "tuning_result": st.session_state.tuning_result,
            "completed_steps": st.session_state.completed_steps,
        }
        if st.session_state.training_result is None:
            st.error("Train a model before generating the report.")
        else:
            if st.session_state.business_insights is None:
                if st.session_state.explanations is None:
                    st.session_state.explanations = explain_model(
                        st.session_state.training_result
                    )
                st.session_state.business_insights = generate_business_insights(
                    df,
                    target,
                    st.session_state.training_result,
                    st.session_state.explanations,
                    st.session_state.quality_score,
                )
            with st.spinner("Generating PDF..."):
                filepath = generate_executive_report(pipeline_state)
                log_step("Executive Report")
                with open(filepath, "rb") as f:
                    st.download_button(
                        "Download Executive Report (PDF)",
                        f,
                        file_name=os.path.basename(filepath),
                        mime="application/pdf",
                    )
                st.success(f"Report saved to `{filepath}`")

st.markdown("---")
b_l, _, b_r = st.columns([1, 3, 1])
with b_l:
    if st.button(
        "← Previous",
        disabled=step_idx == 0,
        use_container_width=True,
        key="btn_prev_bottom",
    ):
        st.session_state.nav_step = STEPS[step_idx - 1]
        st.rerun()
with b_r:
    if st.button(
        "Next →",
        disabled=step_idx >= len(STEPS) - 1,
        use_container_width=True,
        key="btn_next_bottom",
    ):
        st.session_state.nav_step = STEPS[step_idx + 1]
        st.rerun()

st.caption(
    "Built with Streamlit · scikit-learn · Optuna · SHAP · ydata-profiling · Plotly | "
    "Deploy locally with `docker compose up`"
)
