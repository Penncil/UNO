# Estimate NCO-CATE signals, prune and fine-tune Base, then evaluate EASE.
# Run in order using run_simulation.py or simulation_benchmark.ipynb.

# %%   Development-NCO prediction
NCO_LR_C = 1.0
NCO_LR_MAX_ITER = 1000

nco_dict = {}
NCO_select = W_split
np.random.seed(TRAIN_SEED)
for j, i in enumerate(NCO_select):
    W = df[column_names_list[i]].to_numpy()
    lr1 = LogisticRegression(C=NCO_LR_C, max_iter=NCO_LR_MAX_ITER, solver="liblinear")
    lr0 = LogisticRegression(C=NCO_LR_C, max_iter=NCO_LR_MAX_ITER, solver="liblinear")
    lr1.fit(X[A == 1], W[A == 1])
    lr0.fit(X[A == 0], W[A == 0])
    nco_dict[str(i)+"nco1_p"] = test_NCO(lr1, X, W)
    nco_dict[str(i)+"nco0_p"] = test_NCO(lr0, X, W)
    if (j+1) % 10 == 0 or j == len(NCO_select)-1:
        print(f"Development NCO models: {j+1}/{len(NCO_select)}")


# %%   High/low split (including the original 0.08 override)
# ---------- NCO-CATE signal threshold settings ----------
SPLIT_RULE = "original_count"      # Count rule: max//2 - COUNT_OFFSET.
NCO_DIFF_THRESHOLD = None          # None: scan thresholds; a number: manual threshold.
TARGET_HIGH_FRACTION = 0.15        # Target proportion for the automatic threshold scan.
HIGH_FRACTION_BOUNDS = (0.10, 0.20)
COUNT_OFFSET = 3                  # Preserve the original max//2 - 3 count cutoff.
THRESHOLD_GRID_SIZE = 1001         # Scan signal thresholds, not model performance.

estimated_nco_cate_signal = np.column_stack([
    nco_dict[str(i)+"nco1_p"] - nco_dict[str(i)+"nco0_p"] for i in NCO_select
])
abs_nco_cate_signal = np.abs(estimated_nco_cate_signal)
if not np.isfinite(abs_nco_cate_signal).all():
    raise ValueError("Estimated NCO-CATE signals contain nonfinite values.")
if SPLIT_RULE != "original_count":
    raise ValueError("This v0 cell implements the explicitly stated count rule only.")
low_fraction, high_fraction = HIGH_FRACTION_BOUNDS
if not (0 < low_fraction <= TARGET_HIGH_FRACTION <= high_fraction < 1):
    raise ValueError("The target must be within HIGH_FRACTION_BOUNDS, inside (0,1).")

threshold_scan = pd.DataFrame(columns=["signal_threshold", "count_cutoff", "high_fraction"])
if NCO_DIFF_THRESHOLD is None:
    if not isinstance(THRESHOLD_GRID_SIZE, int) or THRESHOLD_GRID_SIZE < 3:
        raise ValueError("THRESHOLD_GRID_SIZE must be an integer >=3.")
    threshold_grid = np.linspace(0.0, float(abs_nco_cate_signal.max()), THRESHOLD_GRID_SIZE)
    scan_rows = []
    for candidate_threshold in threshold_grid:
        candidate_counts = np.sum(abs_nco_cate_signal > candidate_threshold, axis=1)
        candidate_cutoff = int(candidate_counts.max()//2 - COUNT_OFFSET)
        if candidate_cutoff <= 0:
            continue  # A nonpositive cutoff assigns everyone to the high-signal group.
        candidate_high = candidate_counts >= candidate_cutoff
        scan_rows.append({"signal_threshold": float(candidate_threshold),
                          "count_cutoff": candidate_cutoff,
                          "high_fraction": float(candidate_high.mean())})
    threshold_scan = pd.DataFrame(scan_rows, columns=threshold_scan.columns)
    eligible = threshold_scan.loc[
        threshold_scan["high_fraction"].between(low_fraction, high_fraction)
    ].copy()
    if eligible.empty:
        raise ValueError(
            "No scanned signal threshold gives a high proportion in the requested range. "
            "Inspect threshold_scan; increase grid resolution or explicitly revise COUNT_OFFSET. "
            "The split rule has not been changed automatically."
        )
    eligible["distance_to_target"] = np.abs(eligible["high_fraction"] - TARGET_HIGH_FRACTION)
    eligible = eligible.sort_values(["distance_to_target", "signal_threshold"])
    threshold_used = float(eligible.iloc[0]["signal_threshold"])
    print("Automatic signal-threshold calibration (development signals only):")
    display(eligible.head(5).reset_index(drop=True))
else:
    threshold_used = float(NCO_DIFF_THRESHOLD)
    if not np.isfinite(threshold_used) or threshold_used < 0:
        raise ValueError("The manual signal threshold must be finite and nonnegative.")



# Preserve the explicit 0.08 override below, followed by the original count rule.
threshold_used =0.08
I = abs_nco_cate_signal > threshold_used
avg_I = np.sum(I, axis=1)
nco_score = avg_I
nco_cutoff = float(np.max(avg_I)//2 - COUNT_OFFSET)
con = avg_I >= nco_cutoff

if not con.any() or con.all():
    raise ValueError("Degenerate high/low split. Change the threshold explicitly.")
if any(np.unique(A[mask]).size != 2 for mask in (con, ~con)):
    raise ValueError("Both high and low subsets must contain both treatment arms.")
actual_high_fraction = float(con.mean())
if not low_fraction <= actual_high_fraction <= high_fraction:
    print("NOTE: manual threshold gives a high proportion outside HIGH_FRACTION_BOUNDS.")
X_h, X_l = Xt[con], Xt[~con]
A_h, A_l = At[con], At[~con]
print("NCO signal threshold used:", threshold_used, "| count cutoff:", nco_cutoff)
print("High / low counts:", int(con.sum()), int((~con).sum()))
print(f"Actual high proportion: {actual_high_fraction:.2%}")
print("Shapes:", X_h.shape, X_l.shape, A_h.shape, A_l.shape)


# %%   Pruning and fine-tuning settings
ratio_range = [0.35,0.4,0.45,0.5  ]  # Choose using development EASE, not held-out EASE or RR.
n_epochs_finetune = 15  # Preserve the uploaded fine-tuning epoch setting.
finetune_batch_size = 32
FINETUNE_LR = 0.001 #0.0005
FINETUNE_WEIGHT_DECAY = 5e-6
FINETUNE_STEP_SIZE = 10
FINETUNE_LR_GAMMA = 0.5 

if not ratio_range or any(not 0 <= r < 1 for r in ratio_range):
    raise ValueError("Every pruning ratio must be in [0,1).")
if n_epochs_finetune < 0 or finetune_batch_size < 1:
    raise ValueError("Invalid fine-tuning epochs/batch size.")


# %%   Pruning and fine-tuning (explicit loops)
# Read the fitted Base architecture; detect settings changed without retraining.
actual_widths = [layer.out_features for layer in model.stack[:-1]]
if len(set(actual_widths)) != 1:
    raise ValueError("This notebook assumes equal starting width in hidden layers.")
base_config = {"in_N": model.stack[0].in_features, "m": actual_widths[0],
               "depth": len(actual_widths), "dropout": model.dropout.p,
               "ps_floor": model.ps_floor}
print("UNO starts from fitted Base:", base_config)
if (n_hidden, n_depth) != (base_config["m"], base_config["depth"]):
    raise ValueError("n_hidden/n_depth changed after Base fitting. Rerun section 4 first.")
assert all(torch.equal(model.state_dict()[k], v) for k, v in trained.items())

# Keep Base unchanged and start each rerun from the saved RNG state.
torch.set_rng_state(base_rng_state.clone())
model_list, pruning_ps_list, EASE_p_list = [], [], []
mask_list, architecture_rows, candidate_rows, finetune_history = [], [], [], []

for ratio in ratio_range:
    model_p = M_pruned(**base_config)
    model_p.load_state_dict(copy.deepcopy(trained), strict=True)
    model_p.eval()
    with torch.no_grad():
        clone_ps = model_p(Xt).numpy().reshape(-1)
        _, p_list = model_p.pre_act(Xt)
    assert np.array_equal(clone_ps, ps_base)
    assert all(model_p.state_dict()[k].data_ptr() != model.state_dict()[k].data_ptr()
               for k in trained)

    # ---------- high/low mean-squared POST-ReLU activation ratio ----------
    sl = []
    for activations in p_list:
        norm_l = torch.mean(activations[~con]**2, dim=0)
        norm_h = torch.mean(activations[con]**2, dim=0)
        sl.append(norm_h/(norm_l + 1e-12))
    sm = torch.cat([s.unsqueeze(1) for s in sl], dim=1)
    num_to_prune = int(base_config["m"]*base_config["depth"]*ratio)
    _, indices = torch.topk(sm.flatten(), num_to_prune)
    row_indices = indices // sm.size(1)
    col_indices = indices % sm.size(1)

    # ---------- prune incoming weights, bias, and outgoing weights ----------
    masks = {name: torch.ones_like(parameter) for name, parameter in model_p.named_parameters()}
    pruned_per_layer = []
    for layer_id in range(base_config["depth"]):
        removed = row_indices[col_indices == layer_id]
        pruned_per_layer.append(int(len(removed)))
        masks[f"stack.{layer_id}.weight"][removed, :] = 0
        masks[f"stack.{layer_id}.bias"][removed] = 0
        masks[f"stack.{layer_id+1}.weight"][:, removed] = 0
    with torch.no_grad():
        for name, parameter in model_p.named_parameters():
            parameter.mul_(masks[name])
        ps_pruning_only_candidate = model_p(Xt).numpy().reshape(-1).copy()
    print(f"ratio={ratio:.2f}: removed={pruned_per_layer}, total={num_to_prune}")

    # ---------- fine-tuning: direct loop, not a function ----------
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_p.parameters(), lr=FINETUNE_LR,
                           weight_decay=FINETUNE_WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=FINETUNE_STEP_SIZE, gamma=FINETUNE_LR_GAMMA)
    for epoch in range(n_epochs_finetune):
        model_p.train()
        running_loss = 0.0
        permutation = torch.randperm(X_l.size(0))
        for i in range(0, X_l.shape[0], finetune_batch_size):
            ind = permutation[i:i+finetune_batch_size]
            batch_x, batch_a = X_l[ind], A_l[ind]
            out = model_p(batch_x)
            optimizer.zero_grad()
            loss = criterion(out, batch_a)
            loss.backward()
            with torch.no_grad():
                for name, parameter in model_p.named_parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(masks[name])
            optimizer.step()
            with torch.no_grad():
                for name, parameter in model_p.named_parameters():
                    parameter.mul_(masks[name])
            running_loss += loss.item()*len(ind)
        scheduler.step()
        model_p.eval()
        with torch.no_grad():
            candidate_ps = model_p(Xt).numpy().reshape(-1)
        accuracy = float(np.mean((candidate_ps >= 0.5) == A))
        finetune_history.append({"ratio": ratio, "epoch": epoch+1,
                                  "loss_low": running_loss/len(X_l), "accuracy_all": accuracy})
        print(f"  UNO {epoch+1:02d}/{n_epochs_finetune}  loss_low={running_loss/len(X_l):.5f}  "
              f"accuracy_all={accuracy:.4f}")

    # ---------- select ONLY using development NCOs ----------
    model_p.eval()
    with torch.no_grad():
        candidate_ps = model_p(Xt).numpy().reshape(-1)
    EASE_dev, _, _ = evaluate_nco_ease(
        df, column_names_list, W_split, A, ipw_weights(A, candidate_ps, PS_EPS),
    )
    EASE_pruning_dev, _, _ = evaluate_nco_ease(
        df, column_names_list, W_split, A, ipw_weights(A, ps_pruning_only_candidate, PS_EPS),
    )
    model_list.append(copy.deepcopy(model_p.state_dict()))
    mask_list.append(copy.deepcopy(masks))
    pruning_ps_list.append(ps_pruning_only_candidate.copy())
    EASE_p_list.append(EASE_dev)
    architecture_rows.append({"ratio": ratio, "start_widths": actual_widths.copy(),
                               "pruned": pruned_per_layer,
                               "retained": [w-p for w, p in zip(actual_widths, pruned_per_layer)]})
    candidate_rows.append({"ratio": ratio, "pruning_only_dev_EASE": EASE_pruning_dev,
                           "UNO_dev_EASE": EASE_dev})
    print("  development EASE:", EASE_dev)

best = int(np.argmin(EASE_p_list))
ratio = float(ratio_range[best])
model_p.load_state_dict(model_list[best], strict=True)
model_p.eval()
with torch.no_grad():
    ps_uno = model_p(Xt).numpy().reshape(-1).copy()
ps_pruning_only = pruning_ps_list[best].copy()
assert all(torch.equal(model.state_dict()[k], v) for k, v in trained.items())
with torch.no_grad():
    assert np.array_equal(model(Xt).numpy().reshape(-1), ps_base)
for name, parameter in model_p.named_parameters():
    assert torch.count_nonzero(parameter.detach()[mask_list[best][name] == 0]) == 0

print("Selected pruning ratio (development EASE only):", ratio)
display(pd.DataFrame(candidate_rows))
display(pd.DataFrame(architecture_rows))


# %%   Held-out NCO evaluation
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

# Retain the original variable names; all use the same held-out NCO set.
EASE_c_left = float(ease_summary.set_index("method").loc["Unadjusted", "heldout_EASE"])
EASE_logit = float(ease_summary.set_index("method").loc["Logistic PS", "heldout_EASE"])
EASE_left = float(ease_summary.set_index("method").loc["Base NN", "heldout_EASE"])
EASE_p_left = float(ease_summary.set_index("method").loc["UNO", "heldout_EASE"])
