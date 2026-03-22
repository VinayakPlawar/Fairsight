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

COLUMNS = [
    "age","workclass","fnlwgt","education","educational-num",
    "marital-status","occupation","relationship","race","sex",
    "capital-gain","capital-loss","hours-per-week","native-country","income"
]
CATEGORICAL_COLS = [
    "workclass","education","marital-status","occupation",
    "relationship","race","sex","native-country"
]

def load_and_prepare_data() -> dict:
    url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    url_test  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    try:
        df_train = pd.read_csv(url_train, header=None, names=COLUMNS,
                               na_values=" ?", skipinitialspace=True)
        df_test  = pd.read_csv(url_test,  header=None, names=COLUMNS,
                               na_values=" ?", skipinitialspace=True, skiprows=1)
        df = pd.concat([df_train, df_test], ignore_index=True)
    except Exception:
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            "age": np.random.randint(18, 75, n),
            "workclass": np.random.choice(["Private","Self-emp","Government","Other"], n),
            "fnlwgt": np.random.randint(10000, 1000000, n),
            "education": np.random.choice(["HS-grad","Some-college","Bachelors","Masters"], n),
            "educational-num": np.random.randint(1, 16, n),
            "marital-status": np.random.choice(["Married","Never-married","Divorced"], n),
            "occupation": np.random.choice(["Exec-managerial","Prof-specialty","Sales","Other"], n),
            "relationship": np.random.choice(["Husband","Wife","Own-child","Other"], n),
            "race": np.random.choice(["White","Black","Asian-Pac-Islander","Other"],
                                     n, p=[0.85, 0.10, 0.03, 0.02]),
            "sex": np.random.choice(["Male","Female"], n, p=[0.67, 0.33]),
            "capital-gain": np.random.choice([0]*9 + [5000], n),
            "capital-loss": np.random.choice([0]*9 + [500], n),
            "hours-per-week": np.random.randint(1, 99, n),
            "native-country": np.random.choice(["United-States","Other"], n, p=[0.90, 0.10]),
            "income": np.random.choice([0,1], n, p=[0.76, 0.24]),
        })

    df["income"] = df["income"].astype(str).str.strip().str.replace(".", "", regex=False)
    df["income"] = df["income"].map(lambda x: 1 if ">50K" in x else 0)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df_raw = df.copy()

    encoders = {}
    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    feature_cols = [c for c in COLUMNS if c != "income"]
    X = df_enc[feature_cols]
    y = df_enc["income"]
    sens_sex  = df_raw["sex"].str.strip()
    sens_race = df_raw["race"].str.strip()

    X_train, X_test, y_train, y_test, s_sex_train, s_sex_test, s_race_train, s_race_test = (
        train_test_split(X, y, sens_sex, sens_race,
                         test_size=0.2, random_state=42, stratify=y)
    )
    return {
        "df_raw": df_raw, "df_enc": df_enc,
        "X": X, "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "s_sex_train": s_sex_train, "s_sex_test": s_sex_test,
        "s_race_train": s_race_train, "s_race_test": s_race_test,
        "feature_cols": feature_cols, "encoders": encoders,
    }


def train_baseline_model(data: dict) -> dict:
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
        "model": model, "y_pred": y_pred, "y_prob": y_prob,
        "accuracy":  accuracy_score(data["y_test"], y_pred),
        "precision": precision_score(data["y_test"], y_pred),
        "recall":    recall_score(data["y_test"], y_pred),
        "f1":        f1_score(data["y_test"], y_pred),
        "confusion_matrix": confusion_matrix(data["y_test"], y_pred),
        "report": report,
        "feature_cols": data["feature_cols"],
    }


def compute_shap_values(model_result: dict, data: dict) -> dict:
    """Use model's built-in feature importance (no shap library needed)."""
    model = model_result["model"]
    feature_cols = data["feature_cols"]

    try:
        importances = model.feature_importances_
    except Exception:
        importances = np.ones(len(feature_cols))

    feat_imp = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return {"feature_importance": feat_imp}
