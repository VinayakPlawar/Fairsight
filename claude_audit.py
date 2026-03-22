"""claude_audit.py — call Anthropic API to generate a fairness audit report."""

import os
import json


SYSTEM_PROMPT = """You are an expert AI fairness auditor embedded in a loan approval decision system called FairSight. 
You receive structured fairness metrics from a machine learning model and produce a professional, plain-English audit report for non-technical stakeholders — bank managers, policy officers, and regulators.

Your audit report must:
1. Start with an EXECUTIVE SUMMARY (2-3 sentences).
2. Section: BIAS FINDINGS — explain each metric clearly, what it means in real-world terms, and which demographic groups are most disadvantaged.
3. Section: ROOT CAUSE ANALYSIS — explain the likely causes of bias (historical data patterns, feature correlations with protected attributes, societal inequalities encoded in data).
4. Section: REAL-WORLD IMPACT — describe the concrete harm this bias could cause to affected individuals and communities.
5. Section: RECOMMENDED MITIGATION — recommend the most appropriate strategy and explain why. Give a prioritised action plan.
6. Section: SEVERITY RATING — rate overall fairness risk as: Low / Moderate / High / Critical. Justify your rating.
7. Close with a COMPLIANCE NOTE referencing GDPR Article 22 and the EU AI Act's high-risk system requirements.

Always cite specific numbers from the metrics provided. Be direct, actionable, and empathetic. Do not use jargon without explaining it. Use clear section headings."""


def generate_audit_report(fairness_metrics: dict, mitigation_result: dict | None, accuracy: float) -> str:
    """Call Claude API and return the full audit report as a string."""
    import anthropic

    # Build the user message with all metrics
    scalar = fairness_metrics["scalar_metrics"]
    sr_g   = fairness_metrics["selection_rate_gender"]
    sr_r   = fairness_metrics["selection_rate_race"]

    user_content = f"""Please generate a full fairness audit report for our loan approval ML model.

MODEL PERFORMANCE:
- Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)

FAIRNESS METRICS (BEFORE mitigation):
"""
    for metric, val in scalar.items():
        severity = "FAIR" if abs(val) < 0.05 else ("MODERATE BIAS" if abs(val) < 0.10 else "SEVERE BIAS")
        user_content += f"  - {metric}: {val:.4f} [{severity}]\n"

    user_content += "\nSELECTION RATES BY GENDER:\n"
    for group, rate in sr_g.items():
        user_content += f"  - {group}: {rate:.2%} approved\n"

    user_content += "\nSELECTION RATES BY RACE:\n"
    for group, rate in sr_r.items():
        user_content += f"  - {group}: {rate:.2%} approved\n"

    if mitigation_result:
        user_content += "\nFAIRNESS METRICS (AFTER mitigation):\n"
        for metric, val in mitigation_result["after_metrics"].items():
            severity = "FAIR" if abs(val) < 0.05 else ("MODERATE BIAS" if abs(val) < 0.10 else "SEVERE BIAS")
            user_content += f"  - {metric}: {val:.4f} [{severity}]\n"
        user_content += f"  - Accuracy after mitigation: {mitigation_result['accuracy_after']:.3f}\n"

    user_content += "\nPlease generate the complete audit report now."

    # Get API key
    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", None)

    if not api_key:
        raise ValueError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY in Streamlit secrets or as an environment variable."
        )

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}]
    )

    return message.content[0].text
