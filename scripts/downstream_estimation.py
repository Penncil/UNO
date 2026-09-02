import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.utils import resample


def weighted_poisson_log_risk_ratio(propensity, treatment, outcome):
    weights = np.where(treatment == 1, 1 / propensity, 1 / (1 - propensity))
    design = sm.add_constant(pd.Series(treatment, name="treatment"))
    result = sm.GLM(
        outcome, design, family=sm.families.Poisson(), freq_weights=weights
    ).fit()
    return float(result.params["treatment"]), float(result.bse["treatment"])


def _negative_log_likelihood(theta, log_risk_ratios, standard_errors):
    mean, log_sd = theta
    systematic_sd = np.exp(log_sd)
    total_sd = np.sqrt(np.square(standard_errors) + systematic_sd**2)
    return float(-np.sum(norm.logpdf(log_risk_ratios, loc=mean, scale=total_sd)))


def fit_null_distribution(log_risk_ratios, standard_errors):
    """Fit a Gaussian systematic-error distribution to NCO estimates."""
    log_risk_ratios = np.asarray(log_risk_ratios, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    valid = np.isfinite(log_risk_ratios) & np.isfinite(standard_errors) & (standard_errors > 0)
    log_risk_ratios = log_risk_ratios[valid]
    standard_errors = standard_errors[valid]
    if len(log_risk_ratios) == 0:
        raise ValueError("No finite NCO estimates remain for EASE calculation.")

    initial_sd = max(float(np.std(log_risk_ratios)), 1e-6)
    result = minimize(
        _negative_log_likelihood,
        x0=np.asarray([float(np.mean(log_risk_ratios)), np.log(initial_sd)]),
        args=(log_risk_ratios, standard_errors),
        method="L-BFGS-B",
        bounds=[(None, None), (np.log(1e-6), None)],
    )
    if not result.success:
        raise RuntimeError(f"Null-distribution optimization failed: {result.message}")
    return float(result.x[0]), float(np.exp(result.x[1]))


def expected_absolute_systematic_error(mean, standard_deviation):
    """Return E|Z| for Z distributed as Normal(mean, standard_deviation squared)."""
    if standard_deviation <= 1e-12:
        return abs(float(mean))
    z = mean / standard_deviation
    return float(
        standard_deviation * np.sqrt(2 / np.pi) * np.exp(-0.5 * z**2)
        + mean * (2 * norm.cdf(z) - 1)
    )


def evaluate_nco_ease(propensity, treatment, nco_values, nco_indices):
    """Calculate EASE from IPW estimates for a specified, isolated NCO set."""
    nco_indices = np.asarray(nco_indices, dtype=int)
    if len(nco_indices) == 0:
        raise ValueError("The NCO set for EASE calculation is empty.")

    log_risk_ratios = []
    standard_errors = []
    for nco_index in nco_indices:
        estimate, standard_error = weighted_poisson_log_risk_ratio(
            propensity,
            treatment,
            nco_values[:, nco_index],
        )
        log_risk_ratios.append(estimate)
        standard_errors.append(standard_error)

    mean, standard_deviation = fit_null_distribution(log_risk_ratios, standard_errors)
    ease = expected_absolute_systematic_error(mean, standard_deviation)
    return ease, mean, standard_deviation


def estimate_downstream_effect(
    propensity,
    treatment,
    outcome,
    trim_percentiles=(1.0, 99.0),
    bootstrap_replicates=100,
    seed=25,
):
    """Estimate an IPW risk ratio after propensity-score percentile trimming."""
    lower_threshold, upper_threshold = np.percentile(propensity, trim_percentiles)
    keep = (propensity >= lower_threshold) & (propensity <= upper_threshold)
    log_risk_ratio, model_standard_error = weighted_poisson_log_risk_ratio(
        propensity[keep], treatment[keep], outcome[keep]
    )

    bootstrap_log_risk_ratios = []
    kept_indices = np.flatnonzero(keep)
    for bootstrap_index in range(bootstrap_replicates):
        sampled_indices = resample(
            kept_indices,
            replace=True,
            n_samples=len(kept_indices),
            random_state=seed + bootstrap_index,
        )
        estimate, _ = weighted_poisson_log_risk_ratio(
            propensity[sampled_indices], treatment[sampled_indices], outcome[sampled_indices]
        )
        bootstrap_log_risk_ratios.append(estimate)

    bootstrap_standard_error = float(np.std(bootstrap_log_risk_ratios, ddof=1))
    risk_ratio = float(np.exp(log_risk_ratio))
    confidence_interval = np.exp(
        [
            log_risk_ratio - 1.96 * bootstrap_standard_error,
            log_risk_ratio + 1.96 * bootstrap_standard_error,
        ]
    )

    return {
        "n_total": int(len(propensity)),
        "n_after_trimming": int(keep.sum()),
        "trim_lower": float(lower_threshold),
        "trim_upper": float(upper_threshold),
        "risk_ratio": risk_ratio,
        "ci95_lower": float(confidence_interval[0]),
        "ci95_upper": float(confidence_interval[1]),
        "bootstrap_standard_error_log_scale": bootstrap_standard_error,
        "bootstrap_note": "The fitted propensity model is held fixed.",
    }
