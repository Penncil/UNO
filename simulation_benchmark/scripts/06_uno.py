# Original neuron scoring, masked pruning, low-signal fine-tuning, and development-only selection.
# Source-preserving stage; execute with workflow.run_stage in the shared notebook namespace.
# Numerical expressions, parameter values, and statement order are unchanged.

# %% SOURCE_FRAGMENT 32; original zero-based cell 32; id ee059b49
ratio_range = [0.35,0.4,0.45,0.5  ]  # Original executable candidates; held-out NCOs and target RR errors are not used for selection.
n_epochs_finetune = 15 # Original fine-tuning epoch count.
finetune_batch_size = 32
FINETUNE_LR = 0.001 #0.0005
FINETUNE_WEIGHT_DECAY = 5e-6
FINETUNE_STEP_SIZE = 10
FINETUNE_LR_GAMMA = 0.5 

if not ratio_range or any(not 0 <= r < 1 for r in ratio_range):
    raise ValueError("Every pruning ratio must be in [0,1).")
if n_epochs_finetune < 0 or finetune_batch_size < 1:
    raise ValueError("Invalid fine-tuning epochs/batch size.")

# %% SOURCE_FRAGMENT 34; original zero-based cell 34; id 7d233322
# Read the fitted Base architecture rather than relying on possibly stale dimension variables.
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

# Do not modify Base; every rerun starts from the saved Base RNG state.
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
