# Re-estimate NCO signals, regroup, prune, fine-tune, and select using development EASE.
# Source-preserving stage; execute with workflow.run_stage in the shared notebook namespace.
# Numerical expressions, parameter values, and statement order are unchanged.

# %% SOURCE_FRAGMENT 55; original zero-based cell 55; id 2d7e95a4
# ---------- Inherit the original NCO predictive-model settings ----------
NCO25_LR_C = float(NCO_LR_C)
NCO25_LR_MAX_ITER = int(NCO_LR_MAX_ITER)
NCO25_LR_SEED = int(TRAIN_SEED)

nco25_nco_dict = {}
_nco25_numpy_state_before = np.random.get_state()
try:
    np.random.seed(NCO25_LR_SEED)  # Same seed as original NCO fitting; restore the caller RNG on completion.
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

# %% SOURCE_FRAGMENT 57; original zero-based cell 57; id d8a770a2
# ---------- Preserve the rules actually used in the original analysis ----------
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

# %% SOURCE_FRAGMENT 59; original zero-based cell 59; id 675bad55
# ---------- Inherit the original pruning and fine-tuning settings ----------
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
