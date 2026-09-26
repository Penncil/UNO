# Fit development-NCO models and construct the original signal-based groups.
# Source-preserving stage; execute with workflow.run_stage in the shared notebook namespace.
# Numerical expressions, parameter values, and statement order are unchanged.

# %% SOURCE_FRAGMENT 28; original zero-based cell 28; id 138fd1dc
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

# %% SOURCE_FRAGMENT 30; original zero-based cell 30; id b664e59b
# ---------- Original signal-grouping parameters ----------
SPLIT_RULE = "original_count"      # Preserve the count rule: max(counts)//2 - COUNT_OFFSET.
NCO_DIFF_THRESHOLD = None          # Preserve the scan; threshold_used is explicitly overwritten below.
TARGET_HIGH_FRACTION = 0.15        # Original target fraction for the development-only threshold scan.
HIGH_FRACTION_BOUNDS = (0.10, 0.20)
COUNT_OFFSET = 3                  # Preserve the original offset of 3.
THRESHOLD_GRID_SIZE = 1001         # Scan signal thresholds, not model-performance metrics.

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
            continue  # This would classify everyone as high signal; skip this scan candidate.
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



# Preserve the explicit 0.08 override, then apply the original count-based grouping.
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
