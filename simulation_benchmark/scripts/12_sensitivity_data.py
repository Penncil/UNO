# Validate and snapshot the original state, then regenerate selected development NCOs.
# Source-preserving stage; execute with workflow.run_stage in the shared notebook namespace.
# Numerical expressions, parameter values, and statement order are unchanged.

# %% SOURCE_FRAGMENT 51_initialize; original zero-based cell 51; id 879f32e5
import hashlib as _nco25_hashlib

_nco25_required = [
    "np", "pd", "torch", "nn", "optim", "StepLR", "copy", "time", "display",
    "LogisticRegression", "expit", "M_pruned", "test_NCO", "RR",
    "ipw_weights", "evaluate_nco_ease", "df", "column_names_list", "w_cols",
    "W_split", "W_test", "W_matrix", "X", "X64", "Xt", "At", "A", "U", "C",
    "nco_intercepts", "observed_loadings", "hidden_loadings", "nco_specific_beta",
    "model", "trained", "base_rng_state", "ps_base", "Y", "Y_replicates",
    "p0", "p1", "p_factual", "RR_TRUE", "RR_TRIM_QUANTILES", "PS_EPS",
    "MIN_NCO_PREVALENCE", "MIN_EVENTS_PER_ARM", "NCO_LR_C", "NCO_LR_MAX_ITER",
    "TRAIN_SEED", "threshold_used", "COUNT_OFFSET", "SPLIT_RULE", "ratio_range",
    "n_epochs_finetune", "finetune_batch_size", "FINETUNE_LR",
    "FINETUNE_WEIGHT_DECAY", "FINETUNE_STEP_SIZE", "FINETUNE_LR_GAMMA",
]
_nco25_missing = [k for k in _nco25_required if k not in globals()]
if _nco25_missing:
    raise RuntimeError(
        "Run the primary benchmark through repeated outcomes in the same kernel first. Missing: " + str(_nco25_missing)
    )
if RR_TRIM_QUANTILES is not None:
    raise ValueError("This sensitivity analysis requires RR_TRIM_QUANTILES=None to preserve the evaluation cohort.")
if Xt.device.type != "cpu" or next(model.parameters()).device.type != "cpu":
    raise ValueError("This sensitivity analysis preserves the original CPU-only implementation.")

nco25_dev_indices = [int(v) for v in W_split]
nco25_test_indices = [int(v) for v in W_test]
if (len(nco25_dev_indices) != 40 or len(nco25_test_indices) != 40
        or len(set(nco25_dev_indices + nco25_test_indices)) != 80):
    raise ValueError("Expected the original 40 development and 40 held-out NCOs; no new partition is drawn.")
if (not 0 < NCO25_REPLACEMENT_FRACTION <= 1
        or not float(40 * NCO25_REPLACEMENT_FRACTION).is_integer()):
    raise ValueError("The fraction must select an integer number of development NCOs; 0.25 selects 10.")
nco25_n_replace = int(40 * NCO25_REPLACEMENT_FRACTION)
nco25_n = len(A)
nco25_n_y_replicates = len(Y_replicates)
if (Y_replicates.shape != (nco25_n_y_replicates, nco25_n)
        or nco25_n_y_replicates < 2 or not np.isin(Y_replicates, [0, 1]).all()
        or not np.array_equal(Y_replicates[0], Y)):
    raise RuntimeError("Y_replicates is inconsistent with Y. Rerun target-effect and repeated-outcome evaluation.")
if (len(df) != nco25_n or X.shape != X64.shape or len(X) != nco25_n
        or not np.array_equal(X, Xt.detach().cpu().numpy())
        or not np.array_equal(np.asarray(A).reshape(-1, 1), At.detach().cpu().numpy())):
    raise ValueError("Original cohort X/Xt or A/At values are inconsistent.")
if (list(df.columns) != list(column_names_list)
        or not np.array_equal(df[list(w_cols)].to_numpy(), W_matrix)):
    raise ValueError("The NCO columns of the original df do not match W_matrix.")
if not (np.allclose(p1, RR_TRUE*np.asarray(p0), rtol=0, atol=1e-12)
        and np.allclose(p_factual, np.where(A == 1, p1, p0), rtol=0, atol=1e-12)):
    raise RuntimeError("Outcome risks do not match RR_TRUE/A; Y has not been regenerated automatically.")

# Read-only Base model; saved weights and predictions must agree.
nco25_base_state = copy.deepcopy(trained)
if not all(torch.equal(model.state_dict()[k], v) for k, v in nco25_base_state.items()):
    raise RuntimeError("model no longer matches the saved trained Base parameters.")
nco25_base_check = copy.deepcopy(model)
nco25_base_check.eval()
with torch.no_grad():
    nco25_base_ps_check = nco25_base_check(Xt).detach().cpu().numpy().reshape(-1)
if not np.array_equal(nco25_base_ps_check, ps_base):
    raise RuntimeError("ps_base does not match the fitted Base parameters.")
del nco25_base_check

# Modify a new copy only; never write to the original df.
nco25_original_df = df.copy(deep=True)
nco25_df = df.copy(deep=True)
nco25_original_columns = list(column_names_list)
nco25_original_dev_indices = list(W_split)
nco25_original_test_indices = list(W_test)

def _nco25_array_hash(value):
    """Small integrity helper; training and fine-tuning remain explicit loops."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    digest = _nco25_hashlib.sha256()
    digest.update(str((value.shape, str(value.dtype))).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()

nco25_original_hashes = {
    k: _nco25_array_hash(globals()[k])
    for k in ["X", "X64", "Xt", "At", "U", "A", "W_matrix", "Y", "Y_replicates",
              "p0", "p1", "p_factual", "ps_base", "nco_intercepts", "observed_loadings",
              "hidden_loadings", "nco_specific_beta", "base_rng_state"]
}
if "ps_uno" in globals():
    nco25_original_hashes["ps_uno"] = _nco25_array_hash(ps_uno)

print(f"Fixed cohort: n={nco25_n}; existing Y realizations={nco25_n_y_replicates}")
print(f"Replace {nco25_n_replace}/40 development NCOs; retain all 40 held-out NCOs.")
print("Original Base parameters, predictions, data and Y realizations will not be overwritten.")

# %% SOURCE_FRAGMENT 53; original zero-based cell 53; id 204c0aa4
# Separate generators do not advance the original NumPy RNG state.
nco25_rng_selection = np.random.default_rng(
    np.random.SeedSequence([NCO25_SELECTION_SEED, 0x4E434F25, 1])
)
nco25_rng_w = np.random.default_rng(
    np.random.SeedSequence([NCO25_W_SEED, 0x4E434F25, 2])
)
nco25_selected_indices = sorted(
    int(v) for v in nco25_rng_selection.choice(
        nco25_dev_indices, size=nco25_n_replace, replace=False
    )
)
nco25_selected_names = [column_names_list[v] for v in nco25_selected_indices]
nco25_w_position = {name: j for j, name in enumerate(w_cols)}
if any(name not in nco25_w_position for name in nco25_selected_names):
    raise ValueError("Development NCO column names do not match w_cols.")
nco25_selected_j = np.asarray(
    [nco25_w_position[name] for name in nco25_selected_names], dtype=int
)

# New risks use X and the original observed-variable coefficients, not U, V, or A.
nco25_eta_new = (
    np.asarray(nco_intercepts)[None, nco25_selected_j]
    + np.asarray(C)[:, None]*np.asarray(observed_loadings)[None, nco25_selected_j]
    + np.asarray(X64) @ np.asarray(nco_specific_beta)[:, nco25_selected_j]
)
nco25_p_new = expit(nco25_eta_new)
nco25_W_new = nco25_rng_w.binomial(1, nco25_p_new).astype(np.int64)
nco25_W_matrix = np.asarray(W_matrix).copy()
nco25_W_matrix[:, nco25_selected_j] = nco25_W_new
nco25_hidden_loadings = np.asarray(hidden_loadings).copy()
nco25_hidden_loadings[nco25_selected_j] = 0.0
for _nco25_j, _nco25_name in enumerate(nco25_selected_names):
    nco25_df[_nco25_name] = nco25_W_new[:, _nco25_j]

nco25_test_names = [column_names_list[v] for v in nco25_test_indices]
nco25_unchanged_columns = [c for c in column_names_list if c not in nco25_selected_names]
pd.testing.assert_frame_equal(
    nco25_df[nco25_unchanged_columns], nco25_original_df[nco25_unchanged_columns]
)
assert np.array_equal(nco25_df[list(w_cols)].to_numpy(), nco25_W_matrix)

nco25_nco_rows = []
for _nco25_index in nco25_dev_indices:
    _nco25_name = column_names_list[_nco25_index]
    _nco25_j = nco25_w_position[_nco25_name]
    _nco25_w_old = nco25_original_df[_nco25_name].to_numpy()
    _nco25_w_new = nco25_df[_nco25_name].to_numpy()
    _nco25_prev = float(_nco25_w_new.mean())
    _nco25_prev1 = float(_nco25_w_new[A == 1].mean())
    _nco25_prev0 = float(_nco25_w_new[A == 0].mean())
    _nco25_estimable = (
        _nco25_prev >= MIN_NCO_PREVALENCE
        and MIN_EVENTS_PER_ARM <= _nco25_prev1 < 1
        and MIN_EVENTS_PER_ARM <= _nco25_prev0 < 1
    )
    nco25_nco_rows.append({
        "nco": _nco25_name, "df_column_index": _nco25_index,
        "replaced": _nco25_index in nco25_selected_indices,
        "original_U_loading": float(hidden_loadings[_nco25_j]),
        "scenario_U_loading": float(nco25_hidden_loadings[_nco25_j]),
        "prevalence_original": float(_nco25_w_old.mean()),
        "prevalence_scenario": _nco25_prev,
        "prevalence_treated": _nco25_prev1, "prevalence_control": _nco25_prev0,
        "estimable": bool(_nco25_estimable),
    })
nco25_nco_audit = pd.DataFrame(nco25_nco_rows)
display(nco25_nco_audit.loc[nco25_nco_audit["replaced"]].reset_index(drop=True))
if not nco25_nco_audit["estimable"].all():
    raise ValueError(
        "A development NCO failed the original estimability screen. "
        "No NCO was silently dropped or resampled; inspect nco25_nco_audit."
    )
print("Unchanged: all non-replaced development NCOs and all held-out NCO observations.")
