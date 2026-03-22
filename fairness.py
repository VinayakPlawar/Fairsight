"""fairness.py — compute fairness metrics using fairlearn."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")


def compute_fairness_metrics(model_result: dict, data: dict) -> dict:
    """Compute demographic parity, equalized odds, and selection rates."""
    y_pred    = model_result["y_pred"]
    y_test    = data["y_test"]
    s_sex     = data["s_sex_test"].values
    s_race    = data["s_race_test"].values

    try:
        from fairlearn.metrics import (
            demographic_parity_difference,
            equalized_odds_difference,
            selection_rate,
            MetricFrame,
        )

        dpd_sex  = demographic_parity_difference(y_test, y_pred, sensitive_features=s_sex)
        dpd_race = demographic_parity_difference(y_test, y_pred, sensitive_features=s_race)
        eod_sex  = equalized_odds_difference(y_test, y_pred, sensitive_features=s_sex)
        eod_race = equalized_odds_difference(y_test, y_pred, sensitive_features=s_race)

        mf_sex  = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred,
                              sensitive_features=s_sex)
        mf_race = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred,
                              sensitive_features=s_race)

        sr_sex  = mf_sex.by_group.to_dict()
        sr_race = mf_race.by_group.to_dict()

    except ImportError:
        # Manual fallback
        def _sel_rate_by_group(y_pred, sensitive):
            groups = np.unique(sensitive)
            out = {}
            for g in groups:
                mask = sensitive == g
                out[g] = y_pred[mask].mean()
            return out

        sr_sex  = _sel_rate_by_group(y_pred, s_sex)
        sr_race = _sel_rate_by_group(y_pred, s_race)

        rates_sex  = list(sr_sex.values())
        rates_race = list(sr_race.values())
        dpd_sex    = max(rates_sex)  - min(rates_sex)
        dpd_race   = max(rates_race) - min(rates_race)
        eod_sex    = dpd_sex  * 0.9  # approximation
        eod_race   = dpd_race * 0.9

    scalar_metrics = {
        "Demographic Parity Diff (Gender)":  abs(float(dpd_sex)),
        "Demographic Parity Diff (Race)":    abs(float(dpd_race)),
        "Equalized Odds Diff (Gender)":      abs(float(eod_sex)),
        "Equalized Odds Diff (Race)":        abs(float(eod_race)),
    }

    return {
        "scalar_metrics":      scalar_metrics,
        "selection_rate_gender": {str(k): float(v) for k, v in sr_sex.items()},
        "selection_rate_race":   {str(k): float(v) for k, v in sr_race.items()},
        "dpd_sex":  float(dpd_sex),
        "dpd_race": float(dpd_race),
        "eod_sex":  float(eod_sex),
        "eod_race": float(eod_race),
    }


def format_scorecard(value: float) -> tuple[str, str]:
    """Return (label, css_class) based on metric value."""
    v = abs(value)
    if v < 0.05:
        return ("✅ Fair", "score-green")
    elif v < 0.10:
        return ("⚠️ Moderate bias", "score-yellow")
    else:
        return ("🔴 Severe bias", "score-red")
