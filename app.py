import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FairSight — AI Bias Detection Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #e2e8f0 !important; }
.metric-card {
    background: #1e293b; border-radius: 12px; padding: 20px;
    border-left: 4px solid #3b82f6; margin-bottom: 12px;
}
.score-green  { background:#dcfce7; border-left:4px solid #16a34a; border-radius:8px; padding:12px 16px; }
.score-yellow { background:#fef9c3; border-left:4px solid #ca8a04; border-radius:8px; padding:12px 16px; }
.score-red    { background:#fee2e2; border-left:4px solid #dc2626; border-radius:8px; padding:12px 16px; }
.audit-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:24px; line-height:1.8; }
h1,h2,h3 { color: #0f172a; }
.stButton>button { background:#3b82f6; color:white; border:none; border-radius:8px; padding:10px 24px; font-weight:600; }
.stButton>button:hover { background:#2563eb; }
</style>
""", unsafe_allow_html=True)

# ── Imports from project modules ──────────────────────────────────────────────
from model import load_and_prepare_data, train_baseline_model, compute_shap_values
from fairness import compute_fairness_metrics, format_scorecard
from mitigation import run_mitigation
from claude_audit import generate_audit_report

# ── Session state defaults ─────────────────────────────────────────────────────
for k in ["data_loaded","model_trained","bias_computed","mitigation_done","audit_text"]:
    if k not in st.session_state:
        st.session_state[k] = False

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ FairSight")
    st.markdown("*AI Bias Detection & Mitigation*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Dataset Overview",
        "🤖 Train Baseline Model",
        "🔍 Bias Detection",
        "💡 SHAP Explainability",
        "🛠️ Bias Mitigation",
        "📝 Claude AI Audit Report",
    ])
    st.markdown("---")
    st.markdown("**Status**")
    st.markdown(f"{'✅' if st.session_state.data_loaded  else '⬜'} Data loaded")
    st.markdown(f"{'✅' if st.session_state.model_trained else '⬜'} Model trained")
    st.markdown(f"{'✅' if st.session_state.bias_computed else '⬜'} Bias analysed")
    st.markdown(f"{'✅' if st.session_state.mitigation_done else '⬜'} Mitigation run")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dataset Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("Exploring the **UCI Adult Income** dataset — predicting whether income exceeds $50K/year.")

    with st.spinner("Loading dataset…"):
        data = load_and_prepare_data()
        st.session_state.data = data
        st.session_state.data_loaded = True

    df = data["df_raw"]
    st.success(f"Dataset loaded: **{df.shape[0]:,} rows × {df.shape[1]} columns**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total samples", f"{len(df):,}")
    col2.metric("Features", df.shape[1] - 1)
    col2.metric(">$50K (positive class)", f"{(df['income']==1).sum():,}")

    st.markdown("---")
    st.subheader("Class Balance")
    class_counts = df["income"].value_counts().reset_index()
    class_counts.columns = ["income", "count"]
    class_counts["label"] = class_counts["income"].map({0: "≤$50K", 1: ">$50K"})
    fig = px.bar(class_counts, x="label", y="count", color="label",
                 color_discrete_sequence=["#3b82f6","#10b981"],
                 title="Income class distribution")
    fig.update_layout(showlegend=False, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_g, col_r = st.columns(2)

    with col_g:
        st.subheader("Gender distribution")
        g = df["sex"].value_counts().reset_index()
        g.columns = ["sex","count"]
        fig2 = px.pie(g, names="sex", values="count",
                      color_discrete_sequence=["#6366f1","#f43f5e"])
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.subheader("Race distribution")
        r = df["race"].value_counts().reset_index()
        r.columns = ["race","count"]
        fig3 = px.bar(r, x="race", y="count", color="race",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(showlegend=False, plot_bgcolor="white", xaxis_tickangle=-20)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("Income by Gender")
    gender_income = df.groupby(["sex","income"]).size().reset_index(name="count")
    gender_income["income_label"] = gender_income["income"].map({0:"≤$50K",1:">$50K"})
    fig4 = px.bar(gender_income, x="sex", y="count", color="income_label",
                  barmode="group", color_discrete_sequence=["#3b82f6","#10b981"])
    fig4.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Income by Race")
    race_income = df.groupby(["race","income"]).size().reset_index(name="count")
    race_income["income_label"] = race_income["income"].map({0:"≤$50K",1:">$50K"})
    fig5 = px.bar(race_income, x="race", y="count", color="income_label",
                  barmode="group", color_discrete_sequence=["#3b82f6","#10b981"])
    fig5.update_layout(plot_bgcolor="white", xaxis_tickangle=-20)
    st.plotly_chart(fig5, use_container_width=True)

    with st.expander("Show raw data sample"):
        st.dataframe(df.head(100), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Train Baseline Model
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train Baseline Model":
    st.title("🤖 Train Baseline Model")
    if not st.session_state.data_loaded:
        st.warning("Load the dataset first from **Dataset Overview**.")
        st.stop()

    st.markdown("Training an **XGBoost** classifier on the Adult Income dataset.")

    if st.button("🚀 Train Model"):
        with st.spinner("Training XGBoost… this takes ~20 seconds"):
            result = train_baseline_model(st.session_state.data)
            st.session_state.model_result = result
            st.session_state.model_trained = True
        st.success("Model trained successfully!")

    if st.session_state.model_trained:
        r = st.session_state.model_result
        st.markdown("---")
        st.subheader("Performance Metrics")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",  f"{r['accuracy']:.3f}")
        c2.metric("Precision", f"{r['precision']:.3f}")
        c3.metric("Recall",    f"{r['recall']:.3f}")
        c4.metric("F1 Score",  f"{r['f1']:.3f}")

        st.markdown("---")
        st.subheader("Confusion Matrix")
        cm = r["confusion_matrix"]
        fig = px.imshow(cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["≤$50K","$50K+"], y=["≤$50K","$50K+"],
            color_continuous_scale="Blues",
            text_auto=True)
        fig.update_layout(width=450, height=400)
        st.plotly_chart(fig)

        st.markdown("---")
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(r["report"]).T.round(3), use_container_width=True)
    else:
        st.info("Click **Train Model** to begin.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Bias Detection
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Bias Detection":
    st.title("🔍 Bias Detection")
    if not st.session_state.model_trained:
        st.warning("Train the model first.")
        st.stop()

    with st.spinner("Computing fairness metrics…"):
        fm = compute_fairness_metrics(
            st.session_state.model_result,
            st.session_state.data
        )
        st.session_state.fairness_metrics = fm
        st.session_state.bias_computed = True

    st.markdown("### Fairness Scorecard")
    st.markdown("Color coding: 🟢 Fair (&lt;0.05)  🟡 Moderate (0.05–0.10)  🔴 Severe (&gt;0.10)")

    for metric_name, val in fm["scalar_metrics"].items():
        label, css = format_scorecard(val)
        st.markdown(
            f'<div class="{css}"><b>{metric_name}</b>: {val:.4f} — {label}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Selection Rate by Gender")
    sr_gender = fm["selection_rate_gender"]
    fig = px.bar(
        x=list(sr_gender.keys()), y=list(sr_gender.values()),
        labels={"x":"Gender","y":"Selection rate"},
        color=list(sr_gender.keys()),
        color_discrete_sequence=["#6366f1","#f43f5e"],
        text=[f"{v:.2%}" for v in sr_gender.values()]
    )
    fig.update_layout(showlegend=False, plot_bgcolor="white", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selection Rate by Race")
    sr_race = fm["selection_rate_race"]
    fig2 = px.bar(
        x=list(sr_race.keys()), y=list(sr_race.values()),
        labels={"x":"Race","y":"Selection rate"},
        color=list(sr_race.keys()),
        color_discrete_sequence=px.colors.qualitative.Set2,
        text=[f"{v:.2%}" for v in sr_race.values()]
    )
    fig2.update_layout(showlegend=False, plot_bgcolor="white",
                       yaxis_tickformat=".0%", xaxis_tickangle=-20)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Metric Definitions")
    with st.expander("What do these metrics mean?"):
        st.markdown("""
- **Demographic Parity Difference**: Difference in positive prediction rates between groups.
  A value of 0 means equal rates. Negative means the disadvantaged group is less likely to be predicted positive.
- **Equalized Odds Difference**: Difference in both True Positive Rate and False Positive Rate across groups.
  Combines both types of classification error.
- **Selection Rate**: Fraction of each group predicted as positive (income > $50K).
  Large gaps indicate disparate treatment.
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SHAP Explainability
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 SHAP Explainability":
    st.title("💡 SHAP Explainability")
    if not st.session_state.model_trained:
        st.warning("Train the model first.")
        st.stop()

    with st.spinner("Computing SHAP values… (~30 seconds)"):
        shap_data = compute_shap_values(
            st.session_state.model_result,
            st.session_state.data
        )
        st.session_state.shap_data = shap_data

    st.markdown("**SHAP (SHapley Additive exPlanations)** shows which features drive predictions the most.")

    feat_imp = shap_data["feature_importance"].head(10)
    fig = px.bar(
        feat_imp, x="importance", y="feature", orientation="h",
        title="Top 10 features by mean |SHAP value|",
        color="importance", color_continuous_scale="Blues"
    )
    fig.update_layout(plot_bgcolor="white", yaxis=dict(autorange="reversed"),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Bias Implications of Top Features")

    bias_explanations = {
        "capital-gain":     ("💰 Wealth proxy", "Capital gains reflect wealth accumulation, which is historically skewed by race and gender due to systemic inequalities."),
        "capital-loss":     ("💸 Wealth proxy", "Similar to capital gain — reflects investment activity more common among privileged groups."),
        "educational-num":  ("🎓 Education level", "Educational attainment varies by race/gender due to historical access disparities."),
        "hours-per-week":   ("⏰ Work hours", "Women disproportionately work part-time due to caregiving responsibilities."),
        "age":              ("📅 Age", "Older cohorts had less gender/race diversity in high-paying roles."),
        "occupation":       ("👔 Occupation", "Strongly correlated with gender/race due to occupational segregation."),
        "marital-status":   ("💍 Marital status", "Married men benefit from income pooling; this feature can encode gender bias."),
        "relationship":     ("👨‍👩‍👧 Relationship role", "Husband/wife role is gender-coded and correlates with income in biased ways."),
        "fnlwgt":           ("📊 Census weight", "Sampling weight — low direct bias risk."),
        "workclass":        ("🏢 Work class", "Public vs private sector split has racial composition differences."),
    }

    for _, row in feat_imp.iterrows():
        fname = row["feature"]
        if fname in bias_explanations:
            icon_label, explanation = bias_explanations[fname]
            with st.expander(f"{icon_label} — **{fname}** (SHAP: {row['importance']:.4f})"):
                st.markdown(explanation)
        else:
            with st.expander(f"**{fname}** (SHAP: {row['importance']:.4f})"):
                st.markdown("Feature contribution to model predictions.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Bias Mitigation
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🛠️ Bias Mitigation":
    st.title("🛠️ Bias Mitigation")
    if not st.session_state.bias_computed:
        st.warning("Run Bias Detection first.")
        st.stop()

    strategy = st.radio("Select mitigation strategy", [
        "Reweighing (pre-processing)",
        "Exponentiated Gradient (in-processing)",
        "Threshold Optimizer (post-processing)",
    ])

    st.markdown("""
| Strategy | How it works |
|---|---|
| **Reweighing** | Assigns higher weights to under-represented group/label combinations during training |
| **Exponentiated Gradient** | Trains the model with explicit fairness constraints via adversarial reductions |
| **Threshold Optimizer** | Adjusts decision thresholds per group post-training to equalize rates |
    """)

    if st.button("⚡ Run Mitigation"):
        with st.spinner(f"Applying {strategy}…"):
            mit_result = run_mitigation(
                strategy,
                st.session_state.model_result,
                st.session_state.data,
                st.session_state.fairness_metrics
            )
            st.session_state.mitigation_result = mit_result
            st.session_state.mitigation_done = True
        st.success("Mitigation complete!")

    if st.session_state.mitigation_done:
        mr = st.session_state.mitigation_result
        before = st.session_state.fairness_metrics["scalar_metrics"]
        after  = mr["after_metrics"]

        st.markdown("---")
        st.subheader("Before vs After Comparison")

        metrics = list(before.keys())
        before_vals = [before[m] for m in metrics]
        after_vals  = [after.get(m, 0) for m in metrics]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Before mitigation", x=metrics, y=before_vals,
                             marker_color="#ef4444"))
        fig.add_trace(go.Bar(name="After mitigation",  x=metrics, y=after_vals,
                             marker_color="#22c55e"))
        fig.update_layout(barmode="group", plot_bgcolor="white",
                          yaxis_title="Metric value (lower = fairer)",
                          xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Before mitigation**")
            for m, v in before.items():
                _, css = format_scorecard(v)
                st.markdown(f'<div class="{css}" style="margin-bottom:6px"><b>{m}</b>: {v:.4f}</div>',
                            unsafe_allow_html=True)
        with c2:
            st.markdown("**After mitigation**")
            for m in metrics:
                v = after.get(m, 0)
                _, css = format_scorecard(v)
                st.markdown(f'<div class="{css}" style="margin-bottom:6px"><b>{m}</b>: {v:.4f}</div>',
                            unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Accuracy Trade-off")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy before", f"{st.session_state.model_result['accuracy']:.3f}")
        c2.metric("Accuracy after",  f"{mr['accuracy_after']:.3f}",
                  delta=f"{mr['accuracy_after']-st.session_state.model_result['accuracy']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Claude AI Audit Report
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Claude AI Audit Report":
    st.title("📝 Claude AI Audit Report")
    if not st.session_state.bias_computed:
        st.warning("Run Bias Detection first to generate a report.")
        st.stop()

    st.markdown("Click below to have **Claude** generate a professional, plain-English fairness audit report.")

    if st.button("🧠 Generate Audit Report with Claude"):
        fm = st.session_state.fairness_metrics
        mr = st.session_state.get("mitigation_result", None)
        acc = st.session_state.model_result["accuracy"]

        with st.spinner("Claude is analysing your model's fairness… (~15 seconds)"):
            try:
                report = generate_audit_report(fm, mr, acc)
                st.session_state.audit_text = report
            except Exception as e:
                st.error(f"Claude API error: {e}")
                st.stop()

    if st.session_state.audit_text:
        st.markdown("---")
        st.markdown('<div class="audit-box">' +
                    st.session_state.audit_text.replace("\n","<br>") +
                    '</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.download_button(
            label="⬇️ Download Audit Report (.txt)",
            data=st.session_state.audit_text,
            file_name="fairsight_audit_report.txt",
            mime="text/plain"
        )

        # PDF download via reportlab
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            import io

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    leftMargin=2*cm, rightMargin=2*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()
            story = [Paragraph("FairSight — AI Bias Audit Report", styles["Title"]), Spacer(1, 0.5*cm)]
            for line in st.session_state.audit_text.split("\n"):
                if line.strip():
                    story.append(Paragraph(line, styles["Normal"]))
                    story.append(Spacer(1, 0.2*cm))
            doc.build(story)
            buf.seek(0)

            st.download_button(
                label="⬇️ Download Audit Report (.pdf)",
                data=buf,
                file_name="fairsight_audit_report.pdf",
                mime="application/pdf"
            )
        except ImportError:
            st.info("Install `reportlab` for PDF export: `pip install reportlab`")
    else:
        st.info("Click the button above to generate your audit report.")
