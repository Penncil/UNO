"""Read-only reporting and checkpoint export; never used to select a model.

The functions in this module do not train, reseed, modify scientific parameters,
or replace fresh outputs with archived reference values. Call them only after
an analysis has finished. They are also safe to call from the original notebook
kernel to archive its already-fitted Base model before restarting that kernel.
"""
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS = ["heldout_EASE", "mean_rr", "mae_rr", "rmse_rr"]


def publication_tables(namespace):
    """Format fresh calculations using the five manuscript-table columns."""
    import pandas as pd
    raw = namespace["comparison_with_ec"]
    order = ["Unadjusted", "Logistic PS", "Base NN", "UNO",
             "Logistic PS + EC (NCO-only)", "Base NN + EC (NCO-only)",
             "Oracle true PS (X,U)"]
    names = {"Base NN": "Base Model", "Logistic PS + EC (NCO-only)": "Logistic PS + EC",
             "Base NN + EC (NCO-only)": "Base Model + EC", "Oracle true PS (X,U)": "Oracle: true PS (X, U)"}
    table1 = raw.set_index("method").loc[order, METRICS].reset_index()
    table1["method"] = table1["method"].replace(names)
    rows = [raw.set_index("method").loc["Base NN", METRICS].to_dict(),
            raw.set_index("method").loc["UNO", METRICS].to_dict()]
    rows[0]["method"] = "Base Model"
    rows[1]["method"] = "UNO (0%; original)"
    if "nco_fraction_summary" in namespace:
        source = namespace["nco_fraction_summary"]
        for pct in sorted(source["no_U_percent"].unique()):
            part = source.loc[(source["no_U_percent"]==pct) & (source["method"]!="Base Model")]
            if len(part)!=1:
                raise ValueError(f"Expected one UNO row for replacement fraction {pct}%.")
            item = part.iloc[0][METRICS].to_dict()
            item["method"] = f"UNO ({int(pct)}%)"
            rows.append(item)
    table2 = pd.DataFrame(rows)[["method"]+METRICS]
    return table1, table2


def compare_saved_tables(namespace):
    """Compare fresh results with displayed values from the uploaded notebook.

    Saved references have six decimal places. This is an output diagnostic,
    not a tolerance used in fitting, model selection, or scientific estimates.
    A mismatch is shown as a mismatch; reference values are never substituted.
    """
    import pandas as pd
    rows=[]
    raw=namespace["comparison_with_ec"].set_index("method")
    ref=pd.read_csv(ROOT/"reference/table1_saved_outputs.csv").rename(
        columns={"heldout_EASE_before_EC":"heldout_EASE"})
    for _,row in ref.iterrows():
        method=row["method"]
        for metric in METRICS:
            value=float(raw.loc[method,metric]);saved=float(row[metric])
            rows.append({"table":"primary", "setting":"original", "method":method,
                         "metric":metric,"fresh_value":value,"saved_display":saved,
                         "absolute_difference":abs(value-saved),
                         "matches_saved_display":abs(value-saved)<=5.1e-7})
    if "nco_fraction_summary" in namespace:
        actual=namespace["nco_fraction_summary"]
        reference=pd.read_csv(ROOT/"reference/table2_saved_outputs.csv")
        for _,row in actual.iterrows():
            selected=reference.loc[(reference["no_U_percent"]==row["no_U_percent"])
                                   & (reference["method"]==row["method"])]
            if len(selected)!=1:
                continue
            for metric in METRICS:
                value=float(row[metric]);saved=float(selected.iloc[0][metric])
                rows.append({"table":"sensitivity", "setting":f'{int(row["no_U_percent"])}%',
                             "method":row["method"],"metric":metric,
                             "fresh_value":value,"saved_display":saved,
                             "absolute_difference":abs(value-saved),
                             "matches_saved_display":abs(value-saved)<=5.1e-7})
    return pd.DataFrame(rows)


def _json_value(value):
    import numpy as np
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, tuple): return list(value)
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def _array_hash(value):
    import numpy as np
    import torch
    if torch.is_tensor(value): value=value.detach().cpu().numpy()
    a=np.ascontiguousarray(value)
    digest=hashlib.sha256()
    digest.update(str((a.shape,str(a.dtype))).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def save_run(namespace, output_root=None):
    """Archive fresh tables, data, predictions, weights, seeds, and environment.

    Each call creates a new directory. Existing files are never overwritten.
    This export does not alter the original df, arrays, model parameters, or RNG
    states. Full-panel history is exported when it exists in the current kernel.
    """
    import numpy as np
    import pandas as pd
    import torch
    parent = ROOT/"results" if output_root is None else Path(output_root)
    tag=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination=parent/f"run_{tag}_{os.getpid()}"
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"raw").mkdir()
    table1, table2=publication_tables(namespace)
    table1.to_csv(destination/"table1.csv",index=False,float_format="%.17g")
    table2.to_csv(destination/"table2.csv",index=False,float_format="%.17g")
    diagnostic=compare_saved_tables(namespace)
    diagnostic.to_csv(destination/"comparison_with_saved_displays.csv",index=False,float_format="%.17g")
    for name in [
        "comparison_with_ec", "rr_replicates_with_ec", "rr_summary", "rr_replicates",
        "ease_summary", "heldout_nco_estimates", "single_rr_summary", "ec_calibration_parameters",
        "ec_single_rr_summary", "paired_comparisons", "threshold_scan", "nco_fraction_summary",
        "nco25_summary", "nco25_rr_replicates", "nco25_heldout_nco_estimates", "nco25_nco_audit",
        "nco25_signal_summary", "nco25_split_summary", "nco25_candidate_table", "nco25_paired_comparison",
    ]:
        if name in namespace:
            namespace[name].to_csv(destination/"raw"/f"{name}.csv",index=False,float_format="%.17g")
    for name in ["base_history", "finetune_history", "candidate_rows", "architecture_rows", "nco25_finetune_history"]:
        if name in namespace:
            pd.DataFrame(namespace[name]).to_csv(destination/"raw"/f"{name}.csv",index=False,float_format="%.17g")
    data_names=["X", "X64", "U", "A", "W_matrix", "C", "S", "V", "V1_raw", "V1",
                "p0", "p1", "p_factual", "Y", "Y_replicates", "e_true_xu", "ps_logit", "ps_base",
                "ps_uno", "ps_pruning_only", "W_split", "W_test", "nco_intercepts", "observed_loadings",
                "hidden_loadings", "nco_specific_beta", "estimated_nco_cate_signal", "con"]
    arrays={name:np.asarray(namespace[name]) for name in data_names if name in namespace}
    # df['Y'] is the initial outcome in the source, not the final benchmark Y.
    if "df" in namespace: arrays["Y_initial_in_df"]=namespace["df"]["Y"].to_numpy(copy=True)
    np.savez_compressed(destination/"primary_arrays.npz",**arrays)
    np.savez_compressed(destination/"base_rng_state.npz",torch_state=namespace["base_rng_state"].cpu().numpy())
    torch.save(namespace["trained"],destination/"base_weights.pt")
    if "model_p" in namespace: torch.save(namespace["model_p"].state_dict(),destination/"uno_original_weights.pt")
    if "mask_list" in namespace: torch.save(namespace["mask_list"],destination/"original_candidate_masks.pt")
    if "nco25_model" in namespace:
        torch.save(namespace["nco25_model"].state_dict(),destination/"uno_sensitivity_current_weights.pt")
        torch.save(namespace["nco25_candidate_masks"],destination/"sensitivity_current_masks.pt")
        np.savez_compressed(destination/"sensitivity_current_arrays.npz",**{
            name:np.asarray(namespace[name]) for name in
            ["nco25_W_matrix","nco25_hidden_loadings","nco25_estimated_nco_cate_signal",
             "nco25_con","nco25_ps_uno","nco25_rr_estimates","nco25_selected_indices"]
        })
    # The original collector stores summaries and replicate-level results only.
    for pct, result in namespace.get("nco_fraction_results",{}).items():
        sub=destination/f"sensitivity_{int(pct):03d}"
        sub.mkdir()
        for key in ["summary","paired_comparison","rr_replicates"]:
            result[key].to_csv(sub/f"{key}.csv",index=False,float_format="%.17g")
        (sub/"selection.json").write_text(json.dumps({k:result[k] for k in
             ["selected_ncos","selected_ratio","high_fraction"]},indent=2,default=_json_value)+"\n")
    parameters=json.loads((ROOT/"reference/parameter_inventory.json").read_text())
    parameter_names=sorted(set(p["name"] for p in parameters["source_assignments"]))
    actual_parameters={key:namespace[key] for key in parameter_names if key in namespace}
    np_state=np.random.get_state()
    np.savez_compressed(destination/"current_rng_states.npz",
                       torch_state=torch.get_rng_state().cpu().numpy(),numpy_keys=np_state[1])
    generators={name:namespace[name].bit_generator.state for name in
                ["rng_z","rng_a","rng_w","rng_parameters","nco25_rng_selection","nco25_rng_w"] if name in namespace}
    distributions={d.metadata["Name"]:d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    (destination/"installed_packages.txt").write_text("\n".join(f"{k}=={v}" for k,v in sorted(distributions.items(),key=lambda t:t[0].lower()))+"\n")
    manifest={
        "python":platform.python_version(),"platform":platform.platform(),"machine":platform.machine(),
        "versions":namespace.get("versions",{}),"torch_build":torch.__config__.show(),
        "actual_final_parameter_values":actual_parameters,"base_epochs_completed":len(namespace["base_history"]),
        "original_selected_ratio":namespace["ratio"],
        "sensitivity_current_fraction":namespace.get("NCO25_REPLACEMENT_FRACTION"),
        "sensitivity_current_selected_ratio":namespace.get("nco25_selected_ratio"),
        "completed_replacement_percentages":sorted(namespace.get("nco_fraction_results",{})),
        "generator_states":generators,"numpy_global_rng_metadata":[np_state[0],np_state[2],np_state[3],np_state[4]],
        "primary_array_hashes":{k:_array_hash(v) for k,v in arrays.items()},
        "base_weight_hashes":{k:_array_hash(v) for k,v in namespace["trained"].items()},
        "matches_saved_notebook_displays":bool(diagnostic["matches_saved_display"].all()),
        "scope":"One fixed cohort; outcome-only repetitions. All exports are fresh computed values.",
        "source_notebook_sha256":json.loads((ROOT/"reference/source_manifest.json").read_text())["source_sha256"],
    }
    try:
        from threadpoolctl import threadpool_info
        manifest["threadpools"]=threadpool_info()
    except ImportError:
        manifest["threadpools"]="not available"
    (destination/"run_manifest.json").write_text(json.dumps(manifest,indent=2,default=_json_value)+"\n")
    return destination
