# NCO-only calibration of the single-outcome estimates.
# Source-preserving stage; execute with workflow.run_stage in the shared notebook namespace.
# Numerical expressions, parameter values, and statement order are unchanged.

# %% SOURCE_FRAGMENT 40; original zero-based cell 40; id 0329e228
# ============================================================
# 8.1b NCO-only empirical calibration: SINGLE Y only
# Run after sections 7 and 8.1. Section 8.2 is NOT required.
# Keep original held-out EASE; calibrate the current single-Y RR.
# No model training, new Y generation, or repeated-Y results used.
# ============================================================
import numpy as np
import pandas as pd
from scipy.stats import norm
from IPython.display import display

EC_BASE_METHODS = ["Base NN", "Logistic PS"]
EC_ALPHA = 0.05

_required = [
    "df", "column_names_list", "A", "Y", "W_split", "W_test", "PS_EPS",
    "ps_by_method", "heldout_nco_estimates", "ease_summary",
    "single_rr_summary", "rr_keep_masks", "rr_raw_weights", "RR_TRUE",
    "ipw_weights", "evaluate_nco_ease", "RR",
]
_missing = [key for key in _required if key not in globals()]
if _missing:
    raise RuntimeError(f"Run sections 7 and 8.1 first. Missing: {_missing}")
if set(W_split) & set(W_test):
    raise ValueError("Development and held-out NCOs must be disjoint.")
if len(W_split) < 2 or len(W_test) < 2:
    raise ValueError("At least two development and two held-out NCOs are required.")
if not 0 < EC_ALPHA < 1 or not np.isfinite(RR_TRUE) or RR_TRUE <= 0:
    raise ValueError("Invalid EC_ALPHA or RR_TRUE.")
if not EC_BASE_METHODS or len(EC_BASE_METHODS) != len(set(EC_BASE_METHODS)):
    raise ValueError("EC_BASE_METHODS must be nonempty and contain no duplicates.")

# ------------------------------------------------------------
# 1. Check that section 8.1 matches the CURRENT Y and RR_TRUE.
#    Refit only the small outcome GLM for checking, not PS models.
# ------------------------------------------------------------
_ec_A = np.asarray(A).reshape(-1)
_ec_Y = np.asarray(Y).reshape(-1)
if _ec_A.shape != _ec_Y.shape or len(df) != len(_ec_A):
    raise ValueError("A, Y and df must have the same number of rows.")
if single_rr_summary["method"].duplicated().any():
    raise ValueError("Duplicate methods in single_rr_summary; rerun 8.1.")
if set(single_rr_summary["method"]) != set(ps_by_method):
    raise ValueError("The method list changed; rerun sections 7 and 8.1.")
if not np.allclose(single_rr_summary["true_RR"], RR_TRUE, rtol=0, atol=1e-12):
    raise ValueError("RR_TRUE changed; rerun 2.5 and 8.1.")

for _row in single_rr_summary.itertuples(index=False):
    _keep = np.asarray(rr_keep_masks[_row.method], dtype=bool)
    _weights = np.asarray(rr_raw_weights[_row.method], dtype=float).reshape(-1)
    if (_keep.shape != _ec_A.shape or _weights.size != _keep.sum()
            or int(_row.n_used) != int(_keep.sum())):
        raise ValueError(f"Inconsistent RR population for {_row.method}; rerun 8.1.")
    _log_rr, _se = RR(_weights, _ec_A[_keep], _ec_Y[_keep])
    if not (np.isclose(np.exp(_log_rr), _row.RR_hat, rtol=1e-8, atol=1e-10)
            and np.isclose(_se, _row.log_rr_se_model, rtol=1e-8, atol=1e-10)):
        raise RuntimeError(
            f"{_row.method}: single_rr_summary is stale. Rerun 8.1, then this cell."
        )

_ec_ease_rows, _ec_param_rows, _ec_single_rows, _ec_nco_tables = [], [], [], []
_ec_zcrit = norm.ppf(1 - EC_ALPHA / 2)
_ec_source_single = single_rr_summary.set_index("method")

for base_method in EC_BASE_METHODS:
    if base_method not in _ec_source_single.index:
        raise ValueError(f"Method not found in section 8.1: {base_method}")
    ec_method = f"{base_method} + EC (NCO-only)"

    # 2. Fit the Gaussian null using DEVELOPMENT NCOs only.
    weights_full = ipw_weights(_ec_A, ps_by_method[base_method], PS_EPS)
    _, dev_table, null_full = evaluate_nco_ease(
        df, column_names_list, W_split, _ec_A, weights_full
    )
    mu_full, tau_full = float(null_full["mean"]), float(null_full["sd"])

    # 3. Keep raw held-out NCO estimates and EASE unchanged.
    #    Calibrated NCO quantities go into SEPARATE columns.
    test_table = heldout_nco_estimates.loc[
        heldout_nco_estimates["method"] == base_method
    ].copy()
    if (len(test_table) != len(W_test)
            or set(test_table["nco_index"]) != set(W_test)):
        raise ValueError(f"Held-out NCOs do not match W_test for {base_method}.")
    test_table["log_rr_EC"] = test_table["log_rr"] - mu_full
    test_table["se_log_rr_EC"] = np.sqrt(test_table["se_log_rr"]**2 + tau_full**2)
    test_table["p_EC"] = 2 * norm.sf(
        np.abs(test_table["log_rr_EC"] / test_table["se_log_rr_EC"])
    )
    test_table["method"] = ec_method
    _ec_nco_tables.append(test_table)

    source_ease = ease_summary.loc[ease_summary["method"] == base_method]
    if len(source_ease) != 1:
        raise ValueError(f"Expected one original EASE row for {base_method}.")
    ease_row = source_ease.iloc[0].to_dict()
    if int(ease_row["n_heldout_ncos"]) != len(W_test):
        raise ValueError("EASE was calculated on a different NCO set; rerun section 7.")
    ease_row["method"] = ec_method
    _ec_ease_rows.append(ease_row)  # Copy, do not refit held-out EASE.

    # 4. Use exactly the RR population/weights from section 8.1.
    #    If RR was trimmed, fit DEVELOPMENT NCOs on that population.
    keep = np.asarray(rr_keep_masks[base_method], dtype=bool)
    null_rr = null_full
    if not keep.all():
        _, _, null_rr = evaluate_nco_ease(
            df.iloc[np.flatnonzero(keep)], column_names_list, W_split,
            _ec_A[keep], rr_raw_weights[base_method]
        )
    mu_rr, tau_rr = float(null_rr["mean"]), float(null_rr["sd"])
    if not (np.isfinite([mu_rr, tau_rr]).all() and tau_rr >= 0):
        raise ValueError(f"Invalid calibration parameters for {base_method}.")

    # 5. Calibrate ONLY the single-Y estimate from section 8.1.
    single = _ec_source_single.loc[base_method]
    rr_before = float(single["RR_hat"])
    sampling_se = float(single["log_rr_se_model"])
    log_rr_ec = np.log(rr_before) - mu_rr
    se_ec = np.sqrt(sampling_se**2 + tau_rr**2)
    rr_ec = float(np.exp(log_rr_ec))

    _ec_single_rows.append({
        "method": ec_method,
        "n_used": int(keep.sum()), "true_RR": float(RR_TRUE),
        "RR_before_EC": rr_before, "RR_EC": rr_ec,
        "RR_error": rr_ec - RR_TRUE,
        "abs_RR_error": abs(rr_ec - RR_TRUE),
        "log_rr_se_model": sampling_se, "log_rr_se_EC": se_ec,
        "CI_lower_EC": np.exp(log_rr_ec - _ec_zcrit * se_ec),
        "CI_upper_EC": np.exp(log_rr_ec + _ec_zcrit * se_ec),
        "p_EC": 2 * norm.sf(abs(log_rr_ec / se_ec)),  # H0: RR = 1
    })
    _ec_param_rows.append({
        "method": ec_method, "n_development_NCOs": len(W_split),
        "mu_full": mu_full, "tau_full": tau_full,
        "mu_RR": mu_rr, "tau_RR": tau_rr,
        "n_RR_patients": int(keep.sum()),
    })

# ------------------------------------------------------------
# 6. Single-Y outputs; do not overwrite the original results.
# ------------------------------------------------------------
ec_ease_summary = pd.DataFrame(_ec_ease_rows)
ec_calibration_parameters = pd.DataFrame(_ec_param_rows)
ec_single_rr_summary = pd.DataFrame(_ec_single_rows)
ec_heldout_nco_estimates = pd.concat(_ec_nco_tables, ignore_index=True)

_rr_cols = ["method", "n_used", "true_RR", "RR_hat", "RR_error", "abs_RR_error"]
_ec_rr_for_comparison = ec_single_rr_summary.rename(columns={"RR_EC": "RR_hat"})

ec_comparison_single = pd.concat([
    ease_summary.merge(
        single_rr_summary[_rr_cols], on="method", validate="one_to_one"
    ),
    ec_ease_summary.merge(
        _ec_rr_for_comparison[_rr_cols], on="method", validate="one_to_one"
    ),
], ignore_index=True)

# Exact equality: EC inherits its parent's RAW held-out EASE.
for base_method in EC_BASE_METHODS:
    parent = ease_summary.set_index("method").loc[base_method, "heldout_EASE"]
    child = ec_ease_summary.set_index("method").loc[
        f"{base_method} + EC (NCO-only)", "heldout_EASE"
    ]
    if child != parent:
        raise AssertionError(f"Raw EASE changed unexpectedly for {base_method}.")

print("Same current Y: single-run RR and estimation errors (not repeated-Y averages):")
display(ec_comparison_single[[
    "method", "heldout_EASE", "n_used", "true_RR", "RR_hat",
    "RR_error", "abs_RR_error",
]].rename(columns={"heldout_EASE": "heldout_EASE_before_EC"}))

print(f"Same current Y: calibrated intervals ({100*(1-EC_ALPHA):g}%):")
display(ec_single_rr_summary[[
    "method", "RR_before_EC", "RR_EC", "CI_lower_EC", "CI_upper_EC", "p_EC"
]])

print("Development-NCO calibration parameters (log-RR scale):")
display(ec_calibration_parameters)
print("Single-run errors are not Bias/MAE/RMSE across repeated datasets.")
print("CIs retain the original model-based SE and treat calibration parameters as fixed.")
