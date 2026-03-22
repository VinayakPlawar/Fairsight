# ⚖️ FairSight — AI Bias Detection & Mitigation Dashboard

FairSight is a full-stack Streamlit application that detects, explains, and mitigates bias in ML models — using the UCI Adult Income dataset as a demonstration. It integrates the **Anthropic Claude API** to generate plain-English fairness audit reports.

---

## 🚀 Quick Start

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd fairsight
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `aif360` may require extra steps on some platforms. If it fails, the app automatically falls back to manual reweighing.

### 4. Set your Anthropic API key

**Option A — Streamlit secrets (recommended for deployment)**

Create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

**Option B — Environment variable**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."        # Mac/Linux
set ANTHROPIC_API_KEY=sk-ant-...             # Windows
```

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📋 Features

| Page | What it does |
|------|-------------|
| **Dataset Overview** | Loads UCI Adult Income data, shows class balance and protected attribute distributions |
| **Train Baseline Model** | Trains XGBoost, shows accuracy / F1 / confusion matrix |
| **Bias Detection** | Computes Demographic Parity & Equalized Odds differences with color-coded scorecard |
| **SHAP Explainability** | Shows top-10 features by SHAP importance with bias implications |
| **Bias Mitigation** | Three strategies: Reweighing, Exponentiated Gradient, Threshold Optimizer |
| **Claude AI Audit Report** | Generates a full plain-English audit report via Claude API + PDF download |

---

## 🗂️ Project Structure

```
fairsight/
├── app.py              ← Main Streamlit application (all pages)
├── model.py            ← Data loading, XGBoost training, SHAP values
├── fairness.py         ← Fairness metric computation (fairlearn)
├── mitigation.py       ← Three mitigation strategies
├── claude_audit.py     ← Anthropic Claude API integration
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🔧 Tech Stack

- **Frontend/UI:** Streamlit + Plotly
- **ML model:** XGBoost (with scikit-learn fallback)
- **Fairness:** fairlearn, aif360
- **Explainability:** SHAP
- **AI audit:** Anthropic Claude (`claude-sonnet-4-20250514`)
- **PDF export:** ReportLab

---

## 🌐 Deployment (Streamlit Cloud)

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. In **Secrets**, add: `ANTHROPIC_API_KEY = "sk-ant-..."`
4. Deploy!

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `aif360` install fails | The app has a manual fallback — you can skip aif360 |
| Dataset fails to download | The app falls back to synthetic data automatically |
| `SHAP` is slow | Normal — SHAP computes 500-sample approximations |
| Claude API key missing | Set `ANTHROPIC_API_KEY` in env or `.streamlit/secrets.toml` |

---

## 📄 License

MIT License — free to use for hackathons, research, and production.
