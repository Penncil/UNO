# Save the current tables, arrays, software versions, and model weights.
# Run in order using run_simulation.py or simulation_benchmark.ipynb.

# %% Save current results
# ============================================================
# 11. Save current results (no scientific calculation is changed)
# Files are written relative to the current working directory.
# Rerunning this cell overwrites the matching files in results/.
# ============================================================
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_output_tables = {
    "ease_summary": ease_summary,
    "heldout_nco_estimates": heldout_nco_estimates,
    "single_rr_summary": single_rr_summary,
    "single_rr_summary_with_ec": single_rr_summary_with_ec,
    "ec_single_rr_summary": ec_single_rr_summary,
    "ec_calibration_parameters": ec_calibration_parameters,
    "ec_heldout_nco_estimates": ec_heldout_nco_estimates,
    "rr_summary_with_ec": rr_summary_with_ec,
    "rr_replicates_with_ec": rr_replicates_with_ec,
    "comparison_with_ec": comparison_with_ec,
    "paired_comparisons": paired_comparisons,
    "glm_equivalence_check": glm_equivalence_check,
    "base_training_history": pd.DataFrame(base_history),
    "finetuning_history": pd.DataFrame(finetune_history),
    "pruning_candidates": pd.DataFrame(candidate_rows),
    "pruning_architectures": pd.DataFrame(architecture_rows),
    "threshold_scan": threshold_scan,
    "nco_replacement_summary": nco25_summary,
    "nco_replacement_rr_replicates": nco25_rr_replicates,
    "nco_replacement_heldout_estimates": nco25_heldout_nco_estimates,
    "nco_replacement_paired_comparison": nco25_paired_comparison,
    "nco_replacement_audit": nco25_nco_audit,
    "nco_replacement_split": nco25_split_summary,
    "nco_replacement_candidates": nco25_candidate_table,
    "nco_fraction_summary": nco_fraction_summary,
}
for _filename, _table in _output_tables.items():
    _table.to_csv(OUTPUT_DIR / f"{_filename}.csv", index=False)

np.savez_compressed(
    OUTPUT_DIR / "simulation_arrays.npz",
    X=X, X64=X64, U=U, A=A, W_matrix=W_matrix,
    Y=Y, Y_replicates=Y_replicates, p0=p0, p1=p1, p_factual=p_factual,
    C=C, S=S, V=V, V1=V1, e_true_xu=e_true_xu,
    W_split=np.asarray(W_split), W_test=np.asarray(W_test),
    ps_logit=ps_logit, ps_base=ps_base, ps_uno=ps_uno,
    ps_pruning_only=ps_pruning_only,
    rr_estimates=rr_estimates, methods=np.asarray(methods, dtype=str),
    nco_replacement_W=nco25_W_matrix,
    nco_replacement_ps_uno=nco25_ps_uno,
    nco_replacement_indices=np.asarray(nco25_selected_indices),
    nco_replacement_rr_estimates=nco25_rr_estimates,
)
torch.save(model.state_dict(), OUTPUT_DIR / "base_model.pt")
torch.save(model_p.state_dict(), OUTPUT_DIR / "uno_model.pt")
torch.save(nco25_model.state_dict(), OUTPUT_DIR / "nco_replacement_uno_model.pt")
(OUTPUT_DIR / "software_versions.json").write_text(
    json.dumps(versions, indent=2), encoding="utf-8"
)
print("Saved results to:", OUTPUT_DIR.resolve())
