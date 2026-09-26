# Evaluate paired sensitivity results and collect completed fractions.
# Source-preserving stage; execute with workflow.run_stage in the shared notebook namespace.
# Numerical expressions, parameter values, and statement order are unchanged.

# %% SOURCE_FRAGMENT 61; original zero-based cell 61; id 72990dfa
nco25_uno_label = f"UNO ({100*NCO25_REPLACEMENT_FRACTION:g}% development NCOs without U)"
nco25_ps_by_method = {
    "Base Model": np.asarray(ps_base).copy(),
    nco25_uno_label: nco25_ps_uno.copy(),
}
nco25_method_names = list(nco25_ps_by_method)
nco25_ease_rows, nco25_heldout_tables = [], []
nco25_arm1_weights, nco25_arm0_weights = [], []
for _nco25_method, _nco25_ps in nco25_ps_by_method.items():
    _nco25_weights = ipw_weights(A, _nco25_ps, PS_EPS)
    # Read the ORIGINAL held-out NCOs. The appendix has not changed any of them.
    _nco25_ease, _nco25_table, _nco25_null = evaluate_nco_ease(
        df, column_names_list, nco25_test_indices, A, _nco25_weights,
    )
    nco25_ease_rows.append({"method": _nco25_method, "heldout_EASE": float(_nco25_ease)})
    _nco25_table = _nco25_table.copy()
    _nco25_table["method"] = _nco25_method
    nco25_heldout_tables.append(_nco25_table)
    _nco25_w1 = _nco25_weights*(A == 1)
    _nco25_w0 = _nco25_weights*(A == 0)
    nco25_arm1_weights.append(_nco25_w1/_nco25_w1.sum())
    nco25_arm0_weights.append(_nco25_w0/_nco25_w0.sum())

nco25_arm1_weights = np.column_stack(nco25_arm1_weights)
nco25_arm0_weights = np.column_stack(nco25_arm0_weights)
nco25_risk1 = Y_replicates @ nco25_arm1_weights
nco25_risk0 = Y_replicates @ nco25_arm0_weights
if not (np.isfinite(nco25_risk1).all() and np.isfinite(nco25_risk0).all()
        and (nco25_risk1 > 0).all() and (nco25_risk0 > 0).all()):
    raise ValueError("A target-outcome replicate has invalid/zero risk; no replicate was dropped.")
nco25_rr_estimates = nco25_risk1/nco25_risk0
nco25_summary_rows, nco25_replicate_tables = [], []
for _nco25_j, _nco25_method in enumerate(nco25_method_names):
    _nco25_rr = nco25_rr_estimates[:, _nco25_j]
    _nco25_error = _nco25_rr - RR_TRUE
    nco25_summary_rows.append({
        "method": _nco25_method, "n_individuals": nco25_n,
        "n_y_replicates": nco25_n_y_replicates, "true_rr": float(RR_TRUE),
        "mean_rr": float(_nco25_rr.mean()), "mae_rr": float(np.abs(_nco25_error).mean()),
        "rmse_rr": float(np.sqrt(np.mean(_nco25_error**2))),
    })
    nco25_replicate_tables.append(pd.DataFrame({
        "replicate": np.arange(1, nco25_n_y_replicates+1), "method": _nco25_method,
        "rr_hat": _nco25_rr, "true_rr": RR_TRUE, "rr_error": _nco25_error,
        "abs_rr_error": np.abs(_nco25_error), "squared_rr_error": _nco25_error**2,
    }))
    # Same normalized-IPW estimator as the original analysis.
    _nco25_logrr_check, _nco25_se_check = RR(
        ipw_weights(A, nco25_ps_by_method[_nco25_method], PS_EPS), A, Y_replicates[0],
    )
    if not np.isclose(np.exp(_nco25_logrr_check), _nco25_rr[0], rtol=0, atol=1e-8):
        raise AssertionError("Weighted-Poisson versus normalized-IPW RR check failed.")

nco25_summary = pd.DataFrame(nco25_ease_rows).merge(
    pd.DataFrame(nco25_summary_rows), on="method", validate="one_to_one", sort=False,
)
nco25_rr_replicates = pd.concat(nco25_replicate_tables, ignore_index=True)
nco25_heldout_nco_estimates = pd.concat(nco25_heldout_tables, ignore_index=True)
_nco25_base_error = nco25_rr_estimates[:, 0] - RR_TRUE
_nco25_uno_error = nco25_rr_estimates[:, 1] - RR_TRUE
_nco25_base_mse = float(np.mean(_nco25_base_error**2))
nco25_paired_comparison = pd.DataFrame([{
    "comparison": "NCO25 UNO minus Base Model",
    "MAE_difference": float(np.mean(np.abs(_nco25_uno_error)-np.abs(_nco25_base_error))),
    "MSE_difference": float(np.mean(_nco25_uno_error**2-_nco25_base_error**2)),
    "relative_RMSE_reduction": (
        float(1-np.sqrt(np.mean(_nco25_uno_error**2)/_nco25_base_mse))
        if _nco25_base_mse > 0 else np.nan
    ),
    "fraction_UNO_lower_absolute_error": float(np.mean(
        np.abs(_nco25_uno_error) < np.abs(_nco25_base_error)
    )),
}])

# Confirm the Base row agrees with the existing paired-Y result, if available.
if "rr_replicates" in globals():
    _nco25_previous_base = rr_replicates.loc[
        rr_replicates["method"].isin(["Base NN", "Base Model"])
    ].sort_values("replicate")
    if (len(_nco25_previous_base) != nco25_n_y_replicates
            or not np.array_equal(_nco25_previous_base["replicate"].to_numpy(),
                                  np.arange(1, nco25_n_y_replicates+1))
            or not np.allclose(_nco25_previous_base["rr_hat"].to_numpy(),
                               nco25_rr_estimates[:, 0], rtol=0, atol=1e-10)):
        raise RuntimeError("Existing Base RR results are stale/inconsistent; rerun original section 8.2.")

# Verify all original arrays, cohort columns, NCO partition and Base weights survived unchanged.
pd.testing.assert_frame_equal(df, nco25_original_df)
assert list(column_names_list) == nco25_original_columns
assert list(W_split) == nco25_original_dev_indices
assert list(W_test) == nco25_original_test_indices
for _nco25_key, _nco25_hash in nco25_original_hashes.items():
    if _nco25_array_hash(globals()[_nco25_key]) != _nco25_hash:
        raise AssertionError(f"Original object changed: {_nco25_key}")
assert all(torch.equal(model.state_dict()[k], v) for k, v in nco25_base_state.items())
assert all(torch.equal(trained[k], v) for k, v in nco25_base_state.items())
pd.testing.assert_frame_equal(nco25_df[nco25_test_names], df[nco25_test_names])

print(f"True RR={RR_TRUE}; same {nco25_n_y_replicates} paired outcome realizations")
display(nco25_summary[["method", "heldout_EASE", "mean_rr", "mae_rr", "rmse_rr"]].round(6))
display(nco25_paired_comparison.round(6))
print("Original data, fitted Base, original UNO predictions (if present), and Y realizations: unchanged.")
print("One fixed cohort and one specified NCO perturbation; only the original Y realizations are repeated.")

# %% SOURCE_FRAGMENT 62; original zero-based cell 62; id 6c6b4a8e
# Run after each completed NCO-sensitivity evaluation.
# Store the completed result before changing the replacement fraction.
if "nco_fraction_results" not in globals():
    nco_fraction_results = {}

nco_fraction_percent = int(round(100 * NCO25_REPLACEMENT_FRACTION))

# Reject stale results that do not match the current replacement-fraction label.
_nco_fraction_expected = (
    f"UNO ({nco_fraction_percent}% development NCOs without U)"
)
if not nco25_summary["method"].eq(_nco_fraction_expected).any():
    raise RuntimeError(
        "The fraction and result label disagree. Complete the sensitivity stages before collecting results."
    )

# Store copies so the next scenario cannot overwrite earlier summaries.
nco_fraction_results[nco_fraction_percent] = {
    "summary": nco25_summary.copy(deep=True),
    "paired_comparison": nco25_paired_comparison.copy(deep=True),
    "rr_replicates": nco25_rr_replicates.copy(deep=True),
    "selected_ncos": list(nco25_selected_names),
    "selected_ratio": float(nco25_selected_ratio),
    "high_fraction": float(nco25_con.mean()),
}

# Combine all replacement fractions completed in this kernel.
nco_fraction_summary = pd.concat(
    [
        result["summary"].assign(no_U_percent=percent)
        for percent, result in sorted(nco_fraction_results.items())
    ],
    ignore_index=True,
)

display(nco_fraction_summary[[
    "no_U_percent", "method",
    "heldout_EASE", "mean_rr", "mae_rr", "rmse_rr"
]].round(6))
