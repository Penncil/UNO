# Run the retained development-NCO replacement experiment.
# Run in order using run_simulation.py or simulation_benchmark.ipynb.

# %%  NCO replacement: settings and original-state snapshots
# ---------- Additional data-generation settings for NCO replacement ----------
NCO25_REPLACEMENT_FRACTION = 1 #0.1#0.75 #0.5 #0.25
NCO25_SELECTION_SEED = 2026    # Select replacement NCOs without using performance.
NCO25_W_SEED = 2026           # Separate NCO-generation stream; do not use rng_w.

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
        "Run the main analysis through outcome repetitions in the same session. Missing: " + str(_nco25_missing)
    )
if RR_TRIM_QUANTILES is not None:
    raise ValueError("NCO replacement requires RR_TRIM_QUANTILES=None to retain the same evaluation population.")
if Xt.device.type != "cpu" or next(model.parameters()).device.type != "cpu":
    raise ValueError("The NCO replacement experiment uses CPU, as in the main analysis.")

nco25_dev_indices = [int(v) for v in W_split]
nco25_test_indices = [int(v) for v in W_test]
if (len(nco25_dev_indices) != 40 or len(nco25_test_indices) != 40
        or len(set(nco25_dev_indices + nco25_test_indices)) != 80):
    raise ValueError("This experiment requires the original 40 development and 40 held-out NCOs; no new split is made.")
if (not 0 < NCO25_REPLACEMENT_FRACTION <= 1
        or not float(40 * NCO25_REPLACEMENT_FRACTION).is_integer()):
    raise ValueError("The replacement fraction must select a whole number of development NCOs; 0.25 selects 10.")
nco25_n_replace = int(40 * NCO25_REPLACEMENT_FRACTION)
nco25_n = len(A)
nco25_n_y_replicates = len(Y_replicates)
if (Y_replicates.shape != (nco25_n_y_replicates, nco25_n)
        or nco25_n_y_replicates < 2 or not np.isin(Y_replicates, [0, 1]).all()
        or not np.array_equal(Y_replicates[0], Y)):
    raise RuntimeError("Y_replicates does not match the current Y; rerun the single-outcome and repeated-outcome stages.")
if (len(df) != nco25_n or X.shape != X64.shape or len(X) != nco25_n
        or not np.array_equal(X, Xt.detach().cpu().numpy())
        or not np.array_equal(np.asarray(A).reshape(-1, 1), At.detach().cpu().numpy())):
    raise ValueError("The original cohort has inconsistent X/Xt or A/At arrays.")
if (list(df.columns) != list(column_names_list)
        or not np.array_equal(df[list(w_cols)].to_numpy(), W_matrix)):
    raise ValueError("The original NCO columns in df do not match W_matrix.")
if not (np.allclose(p1, RR_TRUE*np.asarray(p0), rtol=0, atol=1e-12)
        and np.allclose(p_factual, np.where(A == 1, p1, p0), rtol=0, atol=1e-12)):
    raise RuntimeError("Current outcome risks do not match RR_TRUE/A; Y is not regenerated automatically.")

# Read the original fitted model without modifying it; verify its predictions.
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

# Modify NCOs only in a separate copy; leave the original df unchanged.
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


# %%   Generate the replacement development NCOs
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

# Compute new risks from X and the original observed coefficients, without U, V, or A.
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


# %%  Fit NCO prediction models for the replacement scenario
# ---------- Inherit the original NCO prediction model settings ----------
NCO25_LR_C = float(NCO_LR_C)
NCO25_LR_MAX_ITER = int(NCO_LR_MAX_ITER)
NCO25_LR_SEED = int(TRAIN_SEED)

nco25_nco_dict = {}
_nco25_numpy_state_before = np.random.get_state()
try:
    np.random.seed(NCO25_LR_SEED)  # Match the original NCO stage; restore the RNG afterward.
    for _nco25_number, _nco25_index in enumerate(nco25_dev_indices):
        _nco25_w = nco25_df[column_names_list[_nco25_index]].to_numpy()
        _nco25_lr1 = LogisticRegression(
            C=NCO25_LR_C, max_iter=NCO25_LR_MAX_ITER, solver="liblinear"
        )
        _nco25_lr0 = LogisticRegression(
            C=NCO25_LR_C, max_iter=NCO25_LR_MAX_ITER, solver="liblinear"
        )
        _nco25_lr1.fit(X[A == 1], _nco25_w[A == 1])
        _nco25_lr0.fit(X[A == 0], _nco25_w[A == 0])
        nco25_nco_dict[str(_nco25_index)+"nco1_p"] = test_NCO(_nco25_lr1, X, _nco25_w)
        nco25_nco_dict[str(_nco25_index)+"nco0_p"] = test_NCO(_nco25_lr0, X, _nco25_w)
        if (_nco25_number+1) % 10 == 0:
            print(f"NCO25: fitted {_nco25_number+1}/40 development NCO model pairs")
finally:
    np.random.set_state(_nco25_numpy_state_before)

nco25_estimated_nco_cate_signal = np.column_stack([
    nco25_nco_dict[str(v)+"nco1_p"] - nco25_nco_dict[str(v)+"nco0_p"]
    for v in nco25_dev_indices
])
nco25_abs_nco_cate_signal = np.abs(nco25_estimated_nco_cate_signal)
if not np.isfinite(nco25_abs_nco_cate_signal).all():
    raise ValueError("Nonfinite estimated NCO-CATE signals.")
nco25_signal_summary = pd.DataFrame({
    "nco": [column_names_list[v] for v in nco25_dev_indices],
    "replaced": [v in nco25_selected_indices for v in nco25_dev_indices],
    "mean_absolute_estimated_NCO_CATE_signal": nco25_abs_nco_cate_signal.mean(axis=0),
})
print("Estimated signals may be nonzero even for NCOs with no direct U dependence.")
display(nco25_signal_summary.groupby("replaced", as_index=False).agg(
    n_ncos=("nco", "size"),
    mean_absolute_estimated_signal=("mean_absolute_estimated_NCO_CATE_signal", "mean"),
))


# %%   High/low split for the replacement scenario
# ---------- Reuse the split rule used by the current main analysis ----------
NCO25_SIGNAL_THRESHOLD = float(threshold_used)
NCO25_COUNT_OFFSET = int(COUNT_OFFSET)
if SPLIT_RULE != "original_count":
    raise ValueError("This appendix preserves the original_count rule only.")
if not np.isfinite(NCO25_SIGNAL_THRESHOLD) or NCO25_SIGNAL_THRESHOLD < 0:
    raise ValueError("Invalid original signal threshold.")

nco25_counts = np.sum(nco25_abs_nco_cate_signal > NCO25_SIGNAL_THRESHOLD, axis=1)
nco25_count_cutoff = int(nco25_counts.max()//2 - NCO25_COUNT_OFFSET)
nco25_con = nco25_counts >= nco25_count_cutoff
if nco25_count_cutoff <= 0 or not nco25_con.any() or nco25_con.all():
    raise ValueError(
        "NCO25 has a degenerate high/low split under the inherited rule. "
        "No threshold or split rule was changed automatically."
    )
if any(np.unique(A[_nco25_mask]).size != 2 for _nco25_mask in (nco25_con, ~nco25_con)):
    raise ValueError("Both high- and low-NCO-signal subsets must contain both treatment arms.")
nco25_X_low = Xt[~nco25_con]
nco25_A_low = At[~nco25_con]
nco25_split_summary = pd.DataFrame([{
    "signal_threshold": NCO25_SIGNAL_THRESHOLD,
    "count_offset": NCO25_COUNT_OFFSET,
    "count_cutoff": nco25_count_cutoff,
    "n_high": int(nco25_con.sum()), "n_low": int((~nco25_con).sum()),
    "high_fraction": float(nco25_con.mean()),
}])
display(nco25_split_summary)
if "HIGH_FRACTION_BOUNDS" in globals():
    if not HIGH_FRACTION_BOUNDS[0] <= nco25_con.mean() <= HIGH_FRACTION_BOUNDS[1]:
        print("NOTE: high fraction is outside the original reference range; rule remains unchanged.")


# %%   Scenario pruning and fine-tuning (explicit loops)
# ---------- Inherit the current main-analysis fine-tuning settings ----------
NCO25_RATIO_RANGE = tuple(float(v) for v in ratio_range)
NCO25_FINETUNE_EPOCHS = int(n_epochs_finetune)
NCO25_FINETUNE_BATCH_SIZE = int(finetune_batch_size)
NCO25_FINETUNE_LR = float(FINETUNE_LR)
NCO25_FINETUNE_WEIGHT_DECAY = float(FINETUNE_WEIGHT_DECAY)
NCO25_FINETUNE_STEP_SIZE = int(FINETUNE_STEP_SIZE)
NCO25_FINETUNE_LR_GAMMA = float(FINETUNE_LR_GAMMA)

if (not NCO25_RATIO_RANGE or any(not 0 <= v < 1 for v in NCO25_RATIO_RANGE)
        or NCO25_FINETUNE_EPOCHS < 0 or NCO25_FINETUNE_BATCH_SIZE < 1):
    raise ValueError("Invalid inherited pruning/fine-tuning settings.")
nco25_widths = [v.out_features for v in model.stack[:-1]]
if len(set(nco25_widths)) != 1:
    raise ValueError("The original top-k layout assumes equal initial hidden-layer widths.")
nco25_base_config = {
    "in_N": model.stack[0].in_features, "m": nco25_widths[0],
    "depth": len(nco25_widths), "dropout": model.dropout.p, "ps_floor": model.ps_floor,
}
print("Inherited Base architecture:", nco25_base_config)
print("Candidate ratios:", NCO25_RATIO_RANGE, "| fine-tuning epochs:", NCO25_FINETUNE_EPOCHS)

nco25_candidate_states, nco25_candidate_masks = [], []
nco25_candidate_rows, nco25_finetune_history = [], []
nco25_start_time = time.perf_counter()
_nco25_torch_state_before = torch.get_rng_state().clone()

# fork_rng restores the caller's Torch RNG even if the loop raises an error.
with torch.random.fork_rng(devices=[]):
    torch.set_rng_state(base_rng_state.clone())
    for _nco25_ratio in NCO25_RATIO_RANGE:
        nco25_model = M_pruned(**nco25_base_config)
        nco25_model.load_state_dict(copy.deepcopy(nco25_base_state), strict=True)
        nco25_model.eval()
        with torch.no_grad():
            _nco25_clone_ps, _nco25_activations = nco25_model.pre_act(Xt)
        assert np.array_equal(_nco25_clone_ps.numpy().reshape(-1), ps_base)
        assert all(nco25_model.state_dict()[k].data_ptr() != model.state_dict()[k].data_ptr()
                   for k in nco25_base_state)

        # Same mean-squared POST-ReLU activation ratio and global top-k layout.
        _nco25_scores = []
        for _nco25_activation in _nco25_activations:
            _nco25_low_ms = torch.mean(_nco25_activation[~nco25_con]**2, dim=0)
            _nco25_high_ms = torch.mean(_nco25_activation[nco25_con]**2, dim=0)
            _nco25_scores.append(_nco25_high_ms/(_nco25_low_ms + 1e-12))
        nco25_neuron_scores = torch.cat([v.unsqueeze(1) for v in _nco25_scores], dim=1)
        if not torch.isfinite(nco25_neuron_scores).all():
            raise ValueError("Nonfinite activation-ratio scores.")
        _nco25_n_pruned = int(nco25_base_config["m"]*nco25_base_config["depth"]*_nco25_ratio)
        _nco25_indices = torch.topk(nco25_neuron_scores.flatten(), _nco25_n_pruned).indices
        _nco25_unit_indices = _nco25_indices // nco25_neuron_scores.size(1)
        _nco25_layer_indices = _nco25_indices % nco25_neuron_scores.size(1)

        # Mask incoming weights, biases, and outgoing weights.
        _nco25_masks = {
            name: torch.ones_like(parameter) for name, parameter in nco25_model.named_parameters()
        }
        _nco25_removed_per_layer = []
        for _nco25_layer in range(nco25_base_config["depth"]):
            _nco25_removed = _nco25_unit_indices[_nco25_layer_indices == _nco25_layer]
            _nco25_removed_per_layer.append(int(len(_nco25_removed)))
            _nco25_masks[f"stack.{_nco25_layer}.weight"][_nco25_removed, :] = 0
            _nco25_masks[f"stack.{_nco25_layer}.bias"][_nco25_removed] = 0
            _nco25_masks[f"stack.{_nco25_layer+1}.weight"][:, _nco25_removed] = 0
        with torch.no_grad():
            for _nco25_name, _nco25_parameter in nco25_model.named_parameters():
                _nco25_parameter.mul_(_nco25_masks[_nco25_name])
        print(f"NCO25 ratio={_nco25_ratio:.2f}; removed per layer={_nco25_removed_per_layer}")

        # Fine-tuning on low-NCO-signal individuals only: explicit loops.
        _nco25_criterion = nn.BCELoss()
        _nco25_optimizer = optim.Adam(
            nco25_model.parameters(), lr=NCO25_FINETUNE_LR,
            weight_decay=NCO25_FINETUNE_WEIGHT_DECAY,
        )
        _nco25_scheduler = StepLR(
            _nco25_optimizer, step_size=NCO25_FINETUNE_STEP_SIZE,
            gamma=NCO25_FINETUNE_LR_GAMMA,
        )
        for _nco25_epoch in range(NCO25_FINETUNE_EPOCHS):
            nco25_model.train()
            _nco25_running_loss = 0.0
            _nco25_permutation = torch.randperm(nco25_X_low.size(0))
            for _nco25_i in range(0, len(nco25_X_low), NCO25_FINETUNE_BATCH_SIZE):
                _nco25_ind = _nco25_permutation[_nco25_i:_nco25_i+NCO25_FINETUNE_BATCH_SIZE]
                _nco25_batch_x = nco25_X_low[_nco25_ind]
                _nco25_batch_a = nco25_A_low[_nco25_ind]
                _nco25_out = nco25_model(_nco25_batch_x)
                _nco25_optimizer.zero_grad()
                _nco25_loss = _nco25_criterion(_nco25_out, _nco25_batch_a)
                _nco25_loss.backward()
                with torch.no_grad():
                    for _nco25_name, _nco25_parameter in nco25_model.named_parameters():
                        if _nco25_parameter.grad is not None:
                            _nco25_parameter.grad.mul_(_nco25_masks[_nco25_name])
                _nco25_optimizer.step()
                with torch.no_grad():
                    for _nco25_name, _nco25_parameter in nco25_model.named_parameters():
                        _nco25_parameter.mul_(_nco25_masks[_nco25_name])
                _nco25_running_loss += _nco25_loss.item()*len(_nco25_ind)
            _nco25_scheduler.step()
            nco25_model.eval()
            with torch.no_grad():
                _nco25_ps_epoch = nco25_model(Xt).numpy().reshape(-1)
            _nco25_accuracy = float(np.mean((_nco25_ps_epoch >= 0.5) == A))
            nco25_finetune_history.append({
                "ratio": _nco25_ratio, "epoch": _nco25_epoch+1,
                "loss_low": _nco25_running_loss/len(nco25_X_low),
                "accuracy_all": _nco25_accuracy,
            })
            print(f"  {_nco25_epoch+1:02d}/{NCO25_FINETUNE_EPOCHS} "
                  f"loss_low={_nco25_running_loss/len(nco25_X_low):.5f} "
                  f"accuracy_all={_nco25_accuracy:.4f}")

        # Selection reads the MODIFIED DEVELOPMENT NCOs, never W_test or Y.
        nco25_model.eval()
        with torch.no_grad():
            _nco25_candidate_ps = nco25_model(Xt).numpy().reshape(-1)
        _nco25_dev_ease, _nco25_dev_table, _nco25_dev_null = evaluate_nco_ease(
            nco25_df, column_names_list, nco25_dev_indices, A,
            ipw_weights(A, _nco25_candidate_ps, PS_EPS),
        )
        nco25_candidate_states.append(copy.deepcopy(nco25_model.state_dict()))
        nco25_candidate_masks.append(copy.deepcopy(_nco25_masks))
        nco25_candidate_rows.append({
            "ratio": _nco25_ratio, "development_EASE": float(_nco25_dev_ease),
            "removed_per_layer": _nco25_removed_per_layer,
            "retained_per_layer": [w-r for w, r in zip(nco25_widths, _nco25_removed_per_layer)],
        })
        print("  development EASE:", _nco25_dev_ease)

    nco25_candidate_table = pd.DataFrame(nco25_candidate_rows)
    nco25_best_index = int(np.argmin(nco25_candidate_table["development_EASE"].to_numpy()))
    nco25_selected_ratio = float(NCO25_RATIO_RANGE[nco25_best_index])
    nco25_model.load_state_dict(nco25_candidate_states[nco25_best_index], strict=True)
    nco25_model.eval()
    with torch.no_grad():
        nco25_ps_uno = nco25_model(Xt).numpy().reshape(-1).copy()
    for _nco25_name, _nco25_parameter in nco25_model.named_parameters():
        assert torch.count_nonzero(
            _nco25_parameter.detach()[nco25_candidate_masks[nco25_best_index][_nco25_name] == 0]
        ) == 0

assert torch.equal(torch.get_rng_state(), _nco25_torch_state_before)
assert all(torch.equal(model.state_dict()[k], v) for k, v in nco25_base_state.items())
nco25_training_seconds = time.perf_counter() - nco25_start_time
print("Selected ratio (development EASE only):", nco25_selected_ratio)
print("NCO25 pruning/fine-tuning seconds:", round(nco25_training_seconds, 2))
display(nco25_candidate_table)


# %%   Evaluate the replacement scenario on the existing outcomes
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


# %%   Accumulate completed replacement scenarios
# Accumulate the scenario after its evaluation has completed.
# Save the completed scenario before changing the replacement fraction.
if "nco_fraction_results" not in globals():
    nco_fraction_results = {}

nco_fraction_percent = int(round(100 * NCO25_REPLACEMENT_FRACTION))

# Reject relabeling old results after changing only the fraction parameter.
_nco_fraction_expected = (
    f"UNO ({nco_fraction_percent}% development NCOs without U)"
)
if not nco25_summary["method"].eq(_nco_fraction_expected).any():
    raise RuntimeError(
        "The current fraction and result label differ; rerun the replacement stage before summarizing."
    )

# Store copies so later nco25_* assignments do not replace completed results.
nco_fraction_results[nco_fraction_percent] = {
    "summary": nco25_summary.copy(deep=True),
    "paired_comparison": nco25_paired_comparison.copy(deep=True),
    "rr_replicates": nco25_rr_replicates.copy(deep=True),
    "selected_ncos": list(nco25_selected_names),
    "selected_ratio": float(nco25_selected_ratio),
    "high_fraction": float(nco25_con.mean()),
}

# Combine all replacement fractions completed in this session.
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

# %%   Display the completed replacement scenarios
#nco_fraction_summary = nco_fraction_summary.iloc[0:0].copy()
display(nco_fraction_summary[[
    "no_U_percent", "method",
    "heldout_EASE", "mean_rr", "mae_rr", "rmse_rr"
]].round(6))
