# Evaluate the unchanged held-out NCO panel.
# Source-preserving stage; execute with workflow.run_stage in the shared notebook namespace.
# Numerical expressions, parameter values, and statement order are unchanged.

# %% SOURCE_FRAGMENT 36; original zero-based cell 36; id a2fe99b9
ps_by_method = {
    "Unadjusted": None,
    "Logistic PS": ps_logit,
    "Base NN": ps_base,
    "Pruning only": ps_pruning_only,
    "UNO": ps_uno,
    "Oracle true PS (X,U)": e_true_xu,
}
ease_rows, nco_tables = [], []
for method, ps in ps_by_method.items():
    weights = ipw_weights(A, ps, PS_EPS)
    ease, table, null = evaluate_nco_ease(df, column_names_list, W_test, A, weights)
    ease_rows.append({"method": method, "n_heldout_ncos": len(W_test),
                      "heldout_EASE": ease, "null_mean": null["mean"], "null_sd": null["sd"]})
    table["method"] = method
    nco_tables.append(table)
ease_summary = pd.DataFrame(ease_rows)
heldout_nco_estimates = pd.concat(nco_tables, ignore_index=True)
display(ease_summary)

# Preserve the original variable names; all evaluations use the same held-out NCOs.
EASE_c_left = float(ease_summary.set_index("method").loc["Unadjusted", "heldout_EASE"])
EASE_logit = float(ease_summary.set_index("method").loc["Logistic PS", "heldout_EASE"])
EASE_left = float(ease_summary.set_index("method").loc["Base NN", "heldout_EASE"])
EASE_p_left = float(ease_summary.set_index("method").loc["UNO", "heldout_EASE"])
