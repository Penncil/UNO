# Apply the final target-outcome specification and estimate single-outcome risk ratios.
# Source-preserving stage; execute with workflow.run_stage in the shared notebook namespace.
# Numerical expressions, parameter values, and statement order are unchanged.

# %% SOURCE_FRAGMENT 38; original zero-based cell 38; id 79a01d57
# Final target-outcome specification from the uploaded notebook; keep this override in place.
Y_GAMMA_U = 2  

Y_USE_V1 =  True       # True uses the quadratic V1 score; False would use the original V.

u1, u2, u3, u4, u5 = np.asarray(U, dtype=float).T

# Use all five unmeasured variables through squares and interactions.
V1_raw = (
    0.5*(u1**2 - 1)
    + 0.4*(u2**2 - 1)
    + 0.3*(u3**2 - 1)
    + 0.2*(u4**2 - 1)
    + 0.5*(u5**2 - 1)#0.1
    + 0.4*(u1*u2 - rho)
    + 0.3*(u3*u4 - rho)
)

# Match the empirical mean and standard deviation within this fixed cohort.
# Do not overwrite V; treatment and the original NCOs still use V.
q_sd = V1_raw.std(ddof=0)
v_sd = np.asarray(V).std(ddof=0)

if not np.isfinite([q_sd, v_sd]).all() or min(q_sd, v_sd) <= 0:
    raise ValueError("V and V1_raw must have positive finite variance.")

V1 = np.mean(V) + v_sd * (V1_raw - V1_raw.mean()) / q_sd

Y_HIDDEN_SCORE = V1 if Y_USE_V1 else V

p0 = expit(
    Y_INTERCEPT + Y_BETA_C*C + Y_GAMMA_U*Y_HIDDEN_SCORE
)



p1 = RR_TRUE*p0
if not (np.isfinite(p0).all() and np.isfinite(p1).all()
        and (p0 > 0).all() and (p0 < 1).all() and (p1 > 0).all() and (p1 < 1).all()):
    raise ValueError("Invalid Y risks. Change the stated DGP; do not clip p1.")
p_factual = np.where(A == 1, p1, p0)
first_y_stream = np.random.SeedSequence([Y_SEED, 0x594F5554]).spawn(1)[0]
Y = np.random.default_rng(first_y_stream).binomial(1, p_factual).astype(np.int64)
assert np.allclose(p1/p0, RR_TRUE)
assert np.isclose(p1.mean()/p0.mean(), RR_TRUE)
print("True marginal RR:", p1.mean()/p0.mean())
print("Mean risks under A=0 / A=1:", p0.mean(), p1.mean())
print("Observed Y events:", int(Y.sum()))

# %% SOURCE_FRAGMENT 39; original zero-based cell 39; id 478051a3
RR_TRIM_QUANTILES = None          # Original setting: no propensity-score trimming.

single_rr_rows, arm1_weights, arm0_weights = [], [], []
rr_raw_weights, rr_keep_masks = {}, {}
methods = list(ps_by_method)
for method, ps in ps_by_method.items():
    keep_mask = np.ones(n, dtype=bool)
    if RR_TRIM_QUANTILES is not None and ps is not None:
        qlo, qhi = RR_TRIM_QUANTILES
        if not 0 <= qlo < qhi <= 1:
            raise ValueError("RR trimming quantiles must satisfy 0 <= low < high <= 1.")
        lower, upper = np.quantile(ps, [qlo, qhi])
        keep_mask = (ps >= lower) & (ps <= upper)
    weights = ipw_weights(A[keep_mask], None if ps is None else ps[keep_mask], PS_EPS)
    log_rr, se = RR(weights, A[keep_mask], Y[keep_mask])
    rr_hat = float(np.exp(log_rr))
    single_rr_rows.append({"method": method, "n_used": int(keep_mask.sum()),
                           "RR_hat": rr_hat, "true_RR": RR_TRUE,
                           "RR_error": rr_hat-RR_TRUE, "abs_RR_error": abs(rr_hat-RR_TRUE),
                           "log_rr_se_model": se})
    full_weights = np.zeros(n)
    full_weights[keep_mask] = weights
    w1, w0 = full_weights*(A == 1), full_weights*(A == 0)
    arm1_weights.append(w1/w1.sum())
    arm0_weights.append(w0/w0.sum())
    rr_raw_weights[method], rr_keep_masks[method] = weights, keep_mask
single_rr_summary = pd.DataFrame(single_rr_rows)
arm1_weights = np.column_stack(arm1_weights)
arm0_weights = np.column_stack(arm0_weights)
display(single_rr_summary)
