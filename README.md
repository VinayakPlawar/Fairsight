# Fairsight
FairSight — AI Bias Detection & Mitigation Dashboard
The Problem
ML models in loan approvals, hiring, and criminal justice encode historical discrimination invisibly. A standard model trained on real income data approves men at 31.1% vs women at 10.8% — a 3x disparity causing real harm at scale, with no visibility into why.
Our Solution
FairSight is a full-stack AI fairness auditing platform that makes bias visible, explainable, and fixable — in one end-to-end pipeline across 6 screens:
① Dataset Overview — Loads UCI Adult Income (48,842 records), visualises income disparities across gender and race instantly.
② Baseline Model — Trains XGBoost (87.3% accuracy) and establishes the biased baseline with full metrics and confusion matrix.
③ Bias Detection — Computes Demographic Parity Difference (0.2031 gender — SEVERE) and Equalized Odds using Microsoft Fairlearn, with a color-coded red/yellow/green scorecard.
④ SHAP Explainability — Ranks top-10 bias-driving features and explains why each carries risk (e.g. hours-per-week disadvantages women due to caregiving).
⑤ Bias Mitigation — Three strategies: Reweighing (IBM AIF360), Exponentiated Gradient, and Threshold Optimizer (Fairlearn). Achieves 60% bias reduction with only 3.2% accuracy trade-off.
⑥ Claude AI Audit — Calls Anthropic Claude (claude-sonnet-4) with all metrics and generates a plain-English, regulator-ready report covering root causes, real-world impact, and recommendations — exported as PDF.
