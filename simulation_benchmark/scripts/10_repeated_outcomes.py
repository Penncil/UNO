# Generate paired outcome replicates, apply fixed calibration, and evaluate accuracy.
# Source-preserving stage; execute with workflow.run_stage in the shared notebook namespace.
# Numerical expressions, parameter values, and statement order are unchanged.

# %% SOURCE_FRAGMENT 42; original zero-based cell 42; id 4e6c8166
N_Y_REPLICATES = 1000    # Preserve all 1,000 paired target-outcome realizations.
if not isinstance(N_Y_REPLICATES, int) or N_Y_REPLICATES < 2:
    raise ValueError("N_Y_REPLICATES must be an integer >= 2.")

y_streams = np.random.SeedSequence([Y_SEED, 0x594F5554]).spawn(N_Y_REPLICATES)
Y_replicates = np.empty((N_Y_REPLICATES, n), dtype=np.uint8)
for b, stream in enumerate(y_streams):
    Y_replicates[b] = np.random.default_rng(stream).binomial(1, p_factual).astype(np.uint8)
assert np.array_equal(Y_replicates[0], Y)

risk1 = Y_replicates @ arm1_weights
risk0 = Y_replicates @ arm0_weights
if not (np.isfinite(risk1).all() and np.isfinite(risk0).all()
        and (risk1 > 0).all() and (risk0 > 0).all()):
    raise ValueError("A repeat has zero events/invalid weighted risks; no repeat is silently dropped.")
rr_estimates = risk1/risk0
log_rr_estimates = np.log(rr_estimates)

check_rows, summary_rows, replicate_tables = [], [], []
for j, method in enumerate(methods):
    poisson_rr = float(single_rr_summary.set_index("method").loc[method, "RR_hat"])
    difference = abs(poisson_rr-rr_estimates[0, j])
    if difference > 1e-8:
        raise AssertionError(f"Weighted-Poisson check failed: {method}, difference={difference}")
    check_rows.append({"method": method, "Poisson_RR": poisson_rr,
                       "weighted_risk_ratio": rr_estimates[0, j], "abs_difference": difference})
    error = rr_estimates[:, j]-RR_TRUE
    log_error = log_rr_estimates[:, j]-np.log(RR_TRUE)
    summary_rows.append({"method": method, "n_y_replicates": N_Y_REPLICATES,
                         "true_rr": RR_TRUE, "mean_rr": rr_estimates[:, j].mean(),
                         "bias_rr": error.mean(), "mae_rr": np.abs(error).mean(),
                         "rmse_rr": np.sqrt(np.mean(error**2)),
                         "empirical_sd_rr": rr_estimates[:, j].std(ddof=1),
                         "bias_log_rr": log_error.mean(),
                         "mae_log_rr": np.abs(log_error).mean(),
                         "rmse_log_rr": np.sqrt(np.mean(log_error**2))})
    replicate_tables.append(pd.DataFrame({
        "replicate": np.arange(1, N_Y_REPLICATES+1), "method": method,
        "rr_hat": rr_estimates[:, j], "true_rr": RR_TRUE, "rr_error": error,
        "abs_rr_error": np.abs(error), "log_rr_error": log_error,
    }))
rr_summary = pd.DataFrame(summary_rows)
rr_replicates = pd.concat(replicate_tables, ignore_index=True)
glm_equivalence_check = pd.DataFrame(check_rows)
comparison = ease_summary.merge(rr_summary, on="method", how="inner")
display(comparison[["method", "heldout_EASE", "mean_rr", "bias_rr", "mae_rr", "rmse_rr"]])
print("Maximum weighted-Poisson check error:", glm_equivalence_check["abs_difference"].max())
print("Outcome-only repeats; independent full datasets = 1.")

# %% SOURCE_FRAGMENT 43; original zero-based cell 43; id 19922671
# ============================================================
# 8.2b Add repeated-Y EC results to the comparison
# Append AFTER the existing 8.2 code.
# First run 8.1 and the SINGLE-Y EC cell after 8.1.
# Reuse mu_RR fitted on development NCOs; do not refit per Y.
# ============================================================
import numpy as np
import pandas as pd
from IPython.display import display

_required = [
    "rr_summary", "rr_replicates", "single_rr_summary", "ease_summary",
    "ec_calibration_parameters", "ec_single_rr_summary", "EC_BASE_METHODS",
    "rr_keep_masks", "RR_TRUE", "N_Y_REPLICATES", "Y", "Y_replicates",
]
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise RuntimeError(f"Run 8.1 -> single-Y EC -> 8.2 first. Missing: {_missing}")
if not EC_BASE_METHODS or len(EC_BASE_METHODS) != len(set(EC_BASE_METHODS)):
    raise ValueError("EC_BASE_METHODS must be nonempty and contain no duplicates.")
if not np.isfinite(RR_TRUE) or RR_TRUE <= 0:
    raise ValueError("RR_TRUE must be finite and positive.")
if not np.array_equal(Y_replicates[0], Y):
    raise RuntimeError("Y changed. Rerun 8.1 -> single-Y EC -> 8.2.")

_ec_params = ec_calibration_parameters.set_index("method", verify_integrity=True)
_ec_single = ec_single_rr_summary.set_index("method", verify_integrity=True)
_single = single_rr_summary.set_index("method", verify_integrity=True)
_raw_summary = rr_summary.set_index("method", verify_integrity=True)
_raw_ease = ease_summary.set_index("method", verify_integrity=True)

_ec_summary_rows, _ec_repeat_tables, _ec_ease_rows = [], [], []

for base_method in EC_BASE_METHODS:
    ec_method = f"{base_method} + EC (NCO-only)"
    if ec_method not in _ec_params.index or ec_method not in _ec_single.index:
        raise RuntimeError(f"Run the single-Y EC cell for {base_method} first.")

    # Same correction as the single-Y EC cell; mu_RR matches RR trimming.
    mu = float(_ec_params.loc[ec_method, "mu_RR"])
    if not np.isfinite(mu):
        raise ValueError(f"Nonfinite mu_RR for {base_method}.")
    if int(_ec_params.loc[ec_method, "n_RR_patients"]) != int(
        np.asarray(rr_keep_masks[base_method], dtype=bool).sum()
    ):
        raise RuntimeError("RR population changed; rerun the single-Y EC cell.")

    base_table = rr_replicates.loc[
        rr_replicates["method"] == base_method
    ].sort_values("replicate").copy()
    if not np.array_equal(
        base_table["replicate"].to_numpy(), np.arange(1, N_Y_REPLICATES + 1)
    ):
        raise ValueError(f"Missing/duplicated repeats for {base_method}.")
    if not np.allclose(base_table["true_rr"], RR_TRUE, rtol=0, atol=1e-12):
        raise RuntimeError("RR_TRUE changed; rerun 8.1, single-Y EC, and 8.2.")

    rr_before = base_table["rr_hat"].to_numpy(dtype=float)
    if not (np.isfinite(rr_before).all() and (rr_before > 0).all()):
        raise ValueError(f"Invalid RR replicates for {base_method}.")
    if not (
        np.isclose(rr_before[0], _single.loc[base_method, "RR_hat"], rtol=0, atol=1e-8)
        and np.isclose(rr_before.mean(), _raw_summary.loc[base_method, "mean_rr"],
                       rtol=0, atol=1e-10)
    ):
        raise RuntimeError("Single-Y and repeated-Y results are not synchronized; rerun 8.2.")

    # Calibrate each paired replicate with ONE fixed NCO-derived shift.
    # tau_RR is for interval calibration, not additional point-estimate noise.
    log_rr_ec = np.log(rr_before) - mu
    rr_ec = np.exp(log_rr_ec)
    if not (np.isfinite(rr_ec).all() and (rr_ec > 0).all()):
        raise ValueError(f"Invalid calibrated RR for {base_method}.")
    if not (
        np.isclose(_ec_single.loc[ec_method, "true_RR"], RR_TRUE, rtol=0, atol=1e-12)
        and np.isclose(rr_ec[0], _ec_single.loc[ec_method, "RR_EC"], rtol=0, atol=1e-8)
    ):
        raise RuntimeError("First EC repeat differs from single-Y EC; rerun that EC cell.")

    error = rr_ec - RR_TRUE
    log_error = log_rr_ec - np.log(RR_TRUE)
    _ec_summary_rows.append({
        "method": ec_method, "n_y_replicates": len(rr_ec), "true_rr": RR_TRUE,
        "mean_rr": rr_ec.mean(), "bias_rr": error.mean(),
        "mae_rr": np.abs(error).mean(), "rmse_rr": np.sqrt(np.mean(error**2)),
        "empirical_sd_rr": rr_ec.std(ddof=1),
        "bias_log_rr": log_error.mean(), "mae_log_rr": np.abs(log_error).mean(),
        "rmse_log_rr": np.sqrt(np.mean(log_error**2)),
    })

    # Keep one calibrated RR per method and replicate, with original pairing.
    base_table["method"] = ec_method
    base_table["rr_hat_before_EC"] = rr_before
    base_table["rr_hat"] = rr_ec
    base_table["rr_error"] = error
    base_table["abs_rr_error"] = np.abs(error)
    base_table["log_rr_error"] = log_error
    _ec_repeat_tables.append(base_table)

    # Copy the parent's RAW held-out EASE exactly; do not recalibrate EASE.
    ease_row = _raw_ease.loc[base_method].to_dict()
    ease_row["method"] = ec_method
    _ec_ease_rows.append(ease_row)

# New combined outputs: original methods/rr_summary/rr_replicates stay unchanged.
ec_rr_summary = pd.DataFrame(_ec_summary_rows)
ec_rr_replicates = pd.concat(_ec_repeat_tables, ignore_index=True)
ec_ease_summary_repeated = pd.DataFrame(_ec_ease_rows)

rr_summary_with_ec = pd.concat(
    [rr_summary, ec_rr_summary], ignore_index=True
)
rr_replicates_with_ec = pd.concat(
    [rr_replicates, ec_rr_replicates], ignore_index=True
)
ease_summary_with_ec = pd.concat(
    [ease_summary, ec_ease_summary_repeated], ignore_index=True
)

comparison_with_ec = rr_summary_with_ec.merge(
    ease_summary_with_ec,
    on="method", how="left", sort=False, validate="one_to_one"
)
if comparison_with_ec["heldout_EASE"].isna().any():
    raise ValueError("Missing original EASE for one or more methods; rerun section 7.")

print("Same Y replicates for all methods; EC uses fixed development-NCO parameters:")
display(comparison_with_ec[[
    "method", "n_y_replicates", "heldout_EASE", "mean_rr",
    "bias_rr", "mae_rr", "rmse_rr",
]].rename(columns={"heldout_EASE": "heldout_EASE_before_EC"}))

print("The first EC replicate matches the single-Y EC result.")
print("EC rows inherit the original held-out EASE of their parent methods.")
print("Outcome-only repeats; independent full datasets = 1.")

# %% SOURCE_FRAGMENT 45; original zero-based cell 45; id 5736fdd6
j_uno = methods.index("UNO")
uno_error = rr_estimates[:, j_uno]-RR_TRUE
paired_rows = []
for baseline in ("Logistic PS", "Base NN"):
    baseline_error = rr_estimates[:, methods.index(baseline)]-RR_TRUE
    paired_rows.append({
        "comparison": f"UNO vs {baseline}",
        "fraction_UNO_lower_abs_error": float(np.mean(np.abs(uno_error) < np.abs(baseline_error))),
        "MAE_difference_UNO_minus_baseline": float(np.mean(np.abs(uno_error)-np.abs(baseline_error))),
        "MSE_difference_UNO_minus_baseline": float(np.mean(uno_error**2-baseline_error**2)),
        "relative_RMSE_reduction": float(1-np.sqrt(np.mean(uno_error**2)/np.mean(baseline_error**2))),
    })
paired_comparisons = pd.DataFrame(paired_rows)
display(paired_comparisons)
