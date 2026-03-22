"""model.py — data loading, XGBoost training, SHAP computation."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")


# ── Column definitions ─────────────────────────────────────────────────────────
COLUMNS = [
    "age","workclass","fnlwgt","education","educational-num",
    "marital-status","occupation","relationship","race","sex",
    "capital-gain","capital-loss","hours-per-week","native-country","income"
]

CATEGORICAL_COLS = [
    "workclass","education","marital-status","occupation",
    "relationship","race","sex","native-country"
]

PROTECTED = ["sex","race"]


def load_and_prepare_data() -> dict:
    """Download the UCI Adult dataset and return prepared splits."""
    url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    url_test  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    try:
        df_train = pd.read_csv(url_train, header=None, names=COLUMNS,
                               na_values=" ?", skipinitialspace=True)
        df_test  = pd.read_csv(url_test,  header=None, names=COLUMNS,
                               na_values=" ?", skipinitialspace=True, skiprows=1)
        df = pd.concat([df_train, df_test], ignore_index=True)
    except Exception:
        # Fallback: generate synthetic data with the right schema
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            "age": np.random.randint(18, 75, n),
            "workclass": np.random.choice(["Private","Self-emp","Government","Other"], n),
            "fnlwgt": np.random.randint(10000, 1000000, n),
            "education": np.random.choice(["HS-grad","Some-college","Bachelors","Masters","Doctorate"], n),
            "educational-num": np.random.randint(1, 16, n),
            "marital-status": np.random.choice(["Married","Never-married","Divorced","Separated"], n),
            "occupation": np.random.choice(["Exec-managerial","Prof-specialty","Craft-repair","Sales","Other"], n),
            "relationship": np.random.choice(["Husband","Wife","Own-child","Not-in-family","Other"], n),
            "race": np.random.choice(["White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"],
                                     n, p=[0.85, 0.10, 0.03, 0.01, 0.01]),
            "sex": np.random.choice(["Male","Female"], n, p=[0.67, 0.33]),
            "capital-gain": np.random.choice([0]*9 + [np.random.randint(1000,50000)], n),
            "capital-loss": np.random.choice([0]*9 + [np.random.randint(100,4000)], n),
            "hours-per-week": np.random.randint(1, 99, n),
            "native-country": np.random.choice(["United-States","Mexico","Other"], n, p=[0.90,0.05,0.05]),
            "income": np.random.choice([0,1], n, p=[0.76, 0.24]),
        })

    # ── Clean income column ──────────────────────────────────────────────────
    df["income"] = df["income"].astype(str).str.strip().str.replace(".", "", regex=False)
    df["income"] = df["income"].map(lambda x: 1 if ">50K" in x else 0)

    # ── Drop missing ──────────────────────────────────────────────────────────
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Keep raw copy before encoding
    df_raw = df.copy()

    # ── Encode categoricals ───────────────────────────────────────────────────
    encoders = {}
    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    feature_cols = [c for c in COLUMNS if c != "income"]
    X = df_enc[feature_cols]
    y = df_enc["income"]

    # Sensitive attributes (raw strings)
    sens_sex  = df_raw["sex"].str.strip()
    sens_race = df_raw["race"].str.strip()

    X_train, X_test, y_train, y_test, s_sex_train, s_sex_test, s_race_train, s_race_test = (
        train_test_split(X, y, sens_sex, sens_race,
                         test_size=0.2, random_state=42, stratify=y)
    )

    return {
        "df_raw":       df_raw,
        "df_enc":       df_enc,
        "X":            X,
        "y":            y,
        "X_train":      X_train,
        "X_test":       X_test,
        "y_train":      y_train,
        "y_test":       y_test,
        "s_sex_train":  s_sex_train,
        "s_sex_test":   s_sex_test,
        "s_race_train": s_race_train,
        "s_race_test":  s_race_test,
        "feature_cols": feature_cols,
        "encoders":     encoders,
    }


def train_baseline_model(data: dict) -> dict:
    """Train XGBoost and return metrics."""
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    model.fit(data["X_train"], data["y_train"])
    y_pred = model.predict(data["X_test"])
    y_prob = model.predict_proba(data["X_test"])[:, 1]

    report = classification_report(data["y_test"], y_pred, output_dict=True)

    return {
        "model":            model,
        "y_pred":           y_pred,
        "y_prob":           y_prob,
        "accuracy":         accuracy_score(data["y_test"], y_pred),
        "precision":        precision_score(data["y_test"], y_pred),
        "recall":           recall_score(data["y_test"], y_pred),
        "f1":               f1_score(data["y_test"], y_pred),
        "confusion_matrix": confusion_matrix(data["y_test"], y_pred),
        "report":           report,
        "feature_cols":     data["feature_cols"],
    }


def compute_shap_values(model_result: dict, data: dict) -> dict:
    """Compute SHAP feature importances."""
    import pandas as pd
    try:
        import shap
        model   = model_result["model"]
        X_test  = data["X_test"]
        sample  = X_test.sample(min(500, len(X_test)), random_state=42)

        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
        except Exception:
            explainer   = shap.Explainer(model, sample)
            sv          = explainer(sample)
            shap_values = sv.values

        if isinstance(shap_values, list):
            sv_arr = shap_values[1]
        else:
            sv_arr = shap_values

        mean_abs = np.abs(sv_arr).mean(axis=0)
        feat_imp = pd.DataFrame({
            "feature":    data["feature_cols"],
            "importance": mean_abs
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    except ImportError:
        # Fallback: use model's built-in feature importance
        model = model_result["model"]
        try:
            importances = model.feature_importances_
        except Exception:
            importances = np.ones(len(data["feature_cols"]))
        feat_imp = pd.DataFrame({
            "feature":    data["feature_cols"],
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    return {"feature_importance": feat_imp}
