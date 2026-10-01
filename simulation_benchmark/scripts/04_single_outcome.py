# Generate Y and evaluate single-outcome RR, confidence intervals, and EC.
# Run in order using run_simulation.py or simulation_benchmark.ipynb.

# %%  Generate the current single outcome Y
# Use the same gamma when comparing the original V and the alternative V1.
Y_GAMMA_U = 2  

Y_USE_V1 =  True       # True: alternative V1; False: original V.

u1, u2, u3, u4, u5 = np.asarray(U, dtype=float).T

# Use all five hidden variables through squared and interaction terms.
V1_raw = (
    0.5*(u1**2 - 1)
    + 0.4*(u2**2 - 1)
    + 0.3*(u3**2 - 1)
    + 0.2*(u4**2 - 1)
    + 0.5*(u5**2 - 1)#0.1
    + 0.4*(u1*u2 - rho)
    + 0.3*(u3*u4 - rho)
)

# Match the mean and standard deviation on this fixed sample.
# Do not overwrite V; treatment and NCO generation still use the original V.
q_sd = V1_raw.std(ddof=0)
v_sd = np.asarray(V).std(ddof=0)

if not np.isfinite([q_sd, v_sd]).all() or min(q_sd, v_sd) <= 0:
    raise ValueError("V and V1_raw must have positive finite variance.")

V1 = np.mean(V) + v_sd * (V1_raw - V1_raw.mean()) / q_sd

Y_HIDDEN_SCORE = V1 if Y_USE_V1 else V

p0 = expit(
    Y_INTERCEPT + Y_BETA_C*C + Y_GAMMA_U*Y_HIDDEN_SCORE
)



p1 = RR_TRUE*p0
if not (np.isfinite(p0).all() and np.isfinite(p1).all()
        and (p0 > 0).all() and (p0 < 1).all() and (p1 > 0).all() and (p1 < 1).all()):
    raise ValueError("Invalid Y risks. Change the stated DGP; do not clip p1.")
p_factual = np.where(A == 1, p1, p0)
first_y_stream = np.random.SeedSequence([Y_SEED, 0x594F5554]).spawn(1)[0]
Y = np.random.default_rng(first_y_stream).binomial(1, p_factual).astype(np.int64)
assert np.allclose(p1/p0, RR_TRUE)
assert np.isclose(p1.mean()/p0.mean(), RR_TRUE)
print("True marginal RR:", p1.mean()/p0.mean())
print("Mean risks under A=0 / A=1:", p0.mean(), p1.mean())
print("Observed Y events:", int(Y.sum()))

# %%   Single-outcome risk ratios
RR_TRIM_QUANTILES = None          # Optional propensity-quantile trimming.

single_rr_rows, arm1_weights, arm0_weights = [], [], []
rr_raw_weights, rr_keep_masks = {}, {}
methods = list(ps_by_method)
for method, ps in ps_by_method.items():
    keep_mask = np.ones(n, dtype=bool)
    if RR_TRIM_QUANTILES is not None and ps is not None:
        qlo, qhi = RR_TRIM_QUANTILES
        if not 0 <= qlo < qhi <= 1:
            raise ValueError("RR trimming quantiles must satisfy 0 <= low < high <= 1.")
        lower, upper = np.quantile(ps, [qlo, qhi])
        keep_mask = (ps >= lower) & (ps <= upper)
    weights = ipw_weights(A[keep_mask], None if ps is None else ps[keep_mask], PS_EPS)
    log_rr, se = RR(weights, A[keep_mask], Y[keep_mask])
    rr_hat = float(np.exp(log_rr))
    single_rr_rows.append({"method": method, "n_used": int(keep_mask.sum()),
                           "RR_hat": rr_hat, "true_RR": RR_TRUE,
                           "RR_error": rr_hat-RR_TRUE, "abs_RR_error": abs(rr_hat-RR_TRUE),
                           "log_rr_se_model": se})
    full_weights = np.zeros(n)
    full_weights[keep_mask] = weights
    w1, w0 = full_weights*(A == 1), full_weights*(A == 0)
    arm1_weights.append(w1/w1.sum())
    arm0_weights.append(w0/w0.sum())
    rr_raw_weights[method], rr_keep_masks[method] = weights, keep_mask
single_rr_summary = pd.DataFrame(single_rr_rows)
arm1_weights = np.column_stack(arm1_weights)
arm0_weights = np.column_stack(arm0_weights)
display(single_rr_summary)


# %%   NCO-only empirical calibration
# ============================================================ 
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
#  Single-Y outputs; do not overwrite the original results.
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


import numpy as np
import pandas as pd
from scipy.stats import norm
from IPython.display import display

CI_ALPHA = 0.05
EC_BASE_METHODS = ["Base NN", "Logistic PS"]

if not 0 < CI_ALPHA < 1:
    raise ValueError("CI_ALPHA must be in (0, 1).")
if set(W_split) & set(W_test):
    raise ValueError("Development and held-out NCOs must be disjoint.")
if len(W_split) < 2:
    raise ValueError("At least two development NCOs are required.")

z_crit = norm.ppf(1 - CI_ALPHA / 2)

# ------------------------------------------------------------
# 1. Uncalibrated methods: intervals use the Poisson model log-RR SE.
# ------------------------------------------------------------
single_rr_summary = single_rr_summary.copy()

rr_values = single_rr_summary["RR_hat"].to_numpy(dtype=float)
se_values = single_rr_summary["log_rr_se_model"].to_numpy(dtype=float)

if not (
    np.isfinite(rr_values).all()
    and (rr_values > 0).all()
    and np.isfinite(se_values).all()
    and (se_values >= 0).all()
):
    raise ValueError("Invalid RR or log-RR SE; rerun 8.1.")

single_rr_summary["log_rr_hat"] = np.log(rr_values)
single_rr_summary["var_log_rr"] = se_values**2
single_rr_summary["se_log_rr"] = se_values

single_rr_summary["LB"] = np.exp(
    single_rr_summary["log_rr_hat"] - z_crit * se_values
)
single_rr_summary["UB"] = np.exp(
    single_rr_summary["log_rr_hat"] + z_crit * se_values
)

# ------------------------------------------------------------
# 2. Fit development NCOs using the same population and weights as the RR.
# ------------------------------------------------------------
base_results = single_rr_summary.set_index("method", verify_integrity=True)
ec_rows, ec_parameter_rows = [], []

for base_method in EC_BASE_METHODS:
    if base_method not in base_results.index:
        raise ValueError(f"Missing method in single_rr_summary: {base_method}")

    ec_method = f"{base_method} + EC (NCO-only)"
    keep = np.asarray(rr_keep_masks[base_method], dtype=bool)
    weights = np.asarray(rr_raw_weights[base_method], dtype=float).reshape(-1)

    if keep.shape != (len(A),) or weights.size != int(keep.sum()):
        raise ValueError(f"Inconsistent RR population or weights: {base_method}")

    # Neither held-out NCOs nor the true outcome effect enter calibration fitting.
    _, _, null_rr = evaluate_nco_ease(
        df.iloc[np.flatnonzero(keep)],
        column_names_list,
        W_split,
        np.asarray(A)[keep],
        weights,
    )
    mu = float(null_rr["mean"])
    tau = float(null_rr["sd"])

    if not (np.isfinite([mu, tau]).all() and tau >= 0):
        raise ValueError(f"Invalid empirical-null parameters: {base_method}")

    base = base_results.loc[base_method]

    log_rr_ec = float(base["log_rr_hat"]) - mu
    sampling_se = float(base["log_rr_se_model"])

    # Add variances on the log-RR scale, not standard errors.
    var_log_rr_ec = sampling_se**2 + tau**2
    se_log_rr_ec = np.sqrt(var_log_rr_ec)

    rr_ec = float(np.exp(log_rr_ec))
    lb_ec = float(np.exp(log_rr_ec - z_crit * se_log_rr_ec))
    ub_ec = float(np.exp(log_rr_ec + z_crit * se_log_rr_ec))

    ec_rows.append({
        "method": ec_method,
        "n_used": int(keep.sum()),
        "true_RR": float(RR_TRUE),
        "RR_before_EC": float(base["RR_hat"]),
        "RR_hat": rr_ec,
        "RR_EC": rr_ec,
        "RR_error": rr_ec - RR_TRUE,
        "abs_RR_error": abs(rr_ec - RR_TRUE),
        "log_rr_hat": log_rr_ec,
        "log_rr_se_model": sampling_se,     # Original model-based sampling SE
        "var_log_rr": var_log_rr_ec,        # EC: sampling variance + tau**2
        "se_log_rr": se_log_rr_ec,
        "log_rr_se_EC": se_log_rr_ec,
        "LB": lb_ec,
        "UB": ub_ec,
        "CI_lower_EC": lb_ec,
        "CI_upper_EC": ub_ec,
    })

    ec_parameter_rows.append({
        "method": ec_method,
        "n_development_NCOs": len(W_split),
        "mu_RR": mu,
        "tau_RR": tau,
        "n_RR_patients": int(keep.sum()),
    })

ec_single_rr_summary = pd.DataFrame(ec_rows)
ec_calibration_parameters = pd.DataFrame(ec_parameter_rows)

# ------------------------------------------------------------
# 3. Combine methods; EC inherits the parent method's uncalibrated EASE. 
# ------------------------------------------------------------
single_rr_summary_with_ec = pd.concat(
    [single_rr_summary, ec_single_rr_summary],
    ignore_index=True,
)

ease_lookup = ease_summary.set_index("method", verify_integrity=True)["heldout_EASE"]
ec_parent = {
    f"{method} + EC (NCO-only)": method
    for method in EC_BASE_METHODS
}

single_rr_summary_with_ec["heldout_EASE"] = [
    float(ease_lookup.loc[ec_parent.get(method, method)])
    for method in single_rr_summary_with_ec["method"]
]

# All rows use the same current Y; LB and UB are on the RR scale.
display(
    single_rr_summary_with_ec[[
        "method",
        "n_used",
        "heldout_EASE",
        "true_RR",
        "log_rr_hat",
        "se_log_rr",
        "RR_hat",
        "LB",
        "UB",
        "RR_error",
        "abs_RR_error",
    ]].rename(columns={"heldout_EASE": "heldout_EASE_before_EC"})
)

# Use RR() model SEs; EC intervals do not additionally account for estimating mu/tau.
