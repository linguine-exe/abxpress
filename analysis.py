import numpy as np
import pandas as pd
import statsmodels.stats.api as sms

from math import erf, sqrt
from statsmodels.stats.weightstats import ttest_ind as sm_ttest_ind

def validate_and_cast(df, variant_col, metric_col, metric_type):
    if variant_col not in df.columns:
        raise ValueError(f"Missing column: {variant_col}")
    if metric_col not in df.columns:
        raise ValueError(f"Missing column: {metric_col}")
    df = df.copy()
    df[variant_col] = df[variant_col].astype(str).str.lower().str.strip()
    if metric_type == "binary":
        df[metric_col] = df[metric_col].astype(int)
        bad = set(df[metric_col].unique()) - {0, 1}
        if bad:
            raise ValueError("Binary metric must be 0 or 1")
    else:
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0)
    return df

def split_groups(df, variant_col, metric_col):
    c = df[df[variant_col] == "control"][metric_col].values
    t = df[df[variant_col] == "treatment"][metric_col].values
    if len(c) == 0 or len(t) == 0:
        raise ValueError("Both control and treatment are required")
    return c, t

def proportion_ztest(control, treatment):
    conv_c = control.sum()
    conv_t = treatment.sum()
    n_c = len(control)
    n_t = len(treatment)
    p_c = conv_c / n_c
    p_t = conv_t / n_t
    pooled = (conv_c + conv_t) / (n_c + n_t)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t)) if pooled not in (0, 1) else 0
    z = (p_t - p_c) / se if se > 0 else 0
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    diff = p_t - p_c
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se
    return {
        "mean_c": p_c, "mean_t": p_t,
        "abs_lift": diff,
        "rel_lift": diff / p_c if p_c != 0 else np.nan,
        "ci_low": ci_low, "ci_high": ci_high,
        "p_value": p_value,
        "n_c": n_c, "n_t": n_t
    }

def ttest_means(control, treatment):
    n_c, n_t = len(control), len(treatment)
    mean_c, mean_t = control.mean(), treatment.mean()
    diff = mean_t - mean_c
    # Welch t-test via statsmodels (equal_var=False)
    tstat, p_value, _ = sm_ttest_ind(treatment, control, usevar="unequal")
    se = np.sqrt(np.var(treatment, ddof=1) / n_t + np.var(control, ddof=1) / n_c)
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se
    rel = diff / mean_c if mean_c != 0 else np.nan
    return {
        "mean_c": mean_c, "mean_t": mean_t,
        "abs_lift": diff,
        "rel_lift": rel,
        "ci_low": ci_low, "ci_high": ci_high,
        "p_value": p_value,
        "n_c": n_c, "n_t": n_t
    }

def estimate_power_binary(p_c, p_t, n_c, n_t, alpha=0.05):
    eff = sms.proportion_effectsize(p_t, p_c)
    analysis = sms.NormalIndPower()
    try:
        return analysis.power(effect_size=eff, nobs1=n_t, ratio=n_c / n_t, alpha=alpha, alternative="two-sided")
    except Exception:
        return np.nan

def estimate_power_means(mean_c, std_c, mean_t, std_t, n_c, n_t, alpha=0.05):
    s = np.sqrt(((std_c ** 2) + (std_t ** 2)) / 2)
    d = (mean_t - mean_c) / s if s > 0 else 0
    analysis = sms.TTestIndPower()
    try:
        return analysis.power(effect_size=d, nobs1=n_t, ratio=n_c / n_t, alpha=alpha, alternative="two-sided")
    except Exception:
        return np.nan

def analyze(df, variant_col, metric_col, metric_type):
    c, t = split_groups(df, variant_col, metric_col)
    if metric_type == "binary":
        res = proportion_ztest(c, t)
        power = estimate_power_binary(res["mean_c"], res["mean_t"], res["n_c"], res["n_t"])
    else:
        res = ttest_means(c, t)
        power = estimate_power_means(
            res["mean_c"], df[df[variant_col] == "control"][metric_col].std(ddof=1),
            res["mean_t"], df[df[variant_col] == "treatment"][metric_col].std(ddof=1),
            res["n_c"], res["n_t"]
        )
    res["power"] = power
    return res

def _norm_cdf(x: float) -> float:
    # Standard normal CDF via error function (no SciPy needed)
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
