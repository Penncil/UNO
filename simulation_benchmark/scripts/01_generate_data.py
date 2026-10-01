# Generate X, U, treatment and NCOs, then fix the NCO split.
# Run in order using run_simulation.py or simulation_benchmark.ipynb.

# %%   Data settings
# ---------- Dimensions and random seeds ----------
DGP_VERSION = "archived_V1_U5"
n = 10000
X_DIM = 20
U_DIM = 5
N_NCO = 80
rho = 0.5
DATA_SEED = 2026
PARAMETER_SEED = 2026  # Keep the 80 NCO parameter sets fixed across data replicates.
NCO_SPLIT_SEED = 2026
Y_SEED = 2026

# ---------- V1 risk functions: C(X), S(X), and V(U) ----------
# Coefficients follow the term order in scripts/functions.py::v1_risk_scores.
C_COEFFICIENTS = np.array([0.8, 0.8, 0.7, 0.6])
S_COEFFICIENTS = np.array([1.0, 0.8, 0.8, 0.6, 0.4])
V_COEFFICIENTS = np.array([0.7, 0.3, 0.2, 0.2])
S_TAIL_WEIGHTS = np.array([1.0, -0.6, 0.8, 0.5, -0.7, 0.6, 0.9])
V_WEIGHTS = np.array([1.0, 0.8, 0.6, 0.5, 0.4])

# ---------- Treatment: G = 1{X1 > 0.5}, with a sigmoid link ----------
GATE_THRESHOLD = 0.5
A_INTERCEPT = -0.2
A_BETA_C = 1.1
A_GAMMA_S = 1.8
A_GAMMA_U = 1.5
OVERLAP_FLOOR = 0.05

# ---------- W ----------
NCO_INTERCEPT_RANGE = (-3.0, -1.5)
NCO_C_RANGE = (0.6, 1.0)
NCO_U_RANGE = (0.9, 1.4)
NCO_SPECIFIC_RANGE = (-0.2, 0.2)
MIN_NCO_PREVALENCE = 0.001
MIN_EVENTS_PER_ARM = 0.001     # A prevalence threshold, not an event count.

# ---------- Outcome: multiplicative risk with a fixed causal RR ----------
RR_TRUE = 0.75
Y_INTERCEPT = -2.5
Y_BETA_C = 0.8 
# ---------- Shared numerical convention for propensity scores ----------
PS_EPS = None                 # Optional numerical clipping; not patient trimming.

if not (isinstance(n, int) and n >= 2 and X_DIM == 20 and U_DIM == 5
        and isinstance(N_NCO, int) and N_NCO >= 4 and abs(rho) < 1):
    raise ValueError("V1 requires X_DIM=20, U_DIM=5; check n, N_NCO and rho.")
if not 0 < OVERLAP_FLOOR < 0.5 or not np.isfinite(RR_TRUE) or RR_TRUE <= 0:
    raise ValueError("Invalid overlap floor or true RR.")


# %%   Joint Gaussian covariates and risk functions
streams = np.random.SeedSequence(DATA_SEED).spawn(3)
rng_z, rng_a, rng_w = [np.random.default_rng(s) for s in streams]
rng_parameters = np.random.default_rng(PARAMETER_SEED)

coordinates = np.arange(X_DIM + U_DIM)
Sigma = rho**np.abs(coordinates[:, None] - coordinates[None, :])
Z = rng_z.standard_normal((n, X_DIM + U_DIM)) @ np.linalg.cholesky(Sigma).T
X64 = Z[:, :X_DIM].copy()
U = Z[:, X_DIM:].copy()

C, S, V = v1_risk_scores(
    X64, U, rho, C_COEFFICIENTS, S_COEFFICIENTS, V_COEFFICIENTS,
    S_TAIL_WEIGHTS, V_WEIGHTS,
)
observed_score, treatment_extra, hidden_score = C, S, V

print("X64, U shapes:", X64.shape, U.shape)
display(pd.DataFrame({
    "score": ["C(X)", "S(X)", "V(U)"],
    "sample_mean": [C.mean(), S.mean(), V.mean()],
    "sample_sd": [C.std(ddof=0), S.std(ddof=0), V.std(ddof=0)],
}))


# %% 2.3 Treatment assignment
gate_score = X64[:, 0]
true_high_region = gate_score > GATE_THRESHOLD
G = true_high_region.astype(float)
eta_a = A_INTERCEPT + A_BETA_C*C + G*(A_GAMMA_S*S + A_GAMMA_U*V)
e_true_xu = OVERLAP_FLOOR + (1-2*OVERLAP_FLOOR)*expit(eta_a)

A = rng_a.binomial(1, e_true_xu).astype(np.int64)
if np.unique(A).size != 2:
    raise ValueError("Only one treatment arm was generated.")
print("A prevalence:", A.mean(), "| true gate proportion:", G.mean())
print("e_true_xu quantiles:", np.quantile(e_true_xu, [0, .01, .5, .99, 1]))


# %% 2.4 Negative-control outcomes
nco_intercepts = rng_parameters.uniform(*NCO_INTERCEPT_RANGE, size=N_NCO)
observed_loadings = rng_parameters.uniform(*NCO_C_RANGE, size=N_NCO)
hidden_loadings = rng_parameters.uniform(*NCO_U_RANGE, size=N_NCO)

nco_specific_beta = np.zeros((X_DIM, N_NCO))
for j in range(N_NCO):
    active = rng_parameters.choice(np.arange(1, 7), size=3, replace=False)  # X2,...,X7
    nco_specific_beta[active, j] = rng_parameters.uniform(*NCO_SPECIFIC_RANGE, size=3)

eta_w = (nco_intercepts[None, :] + C[:, None]*observed_loadings[None, :]
         + V[:, None]*hidden_loadings[None, :] + X64 @ nco_specific_beta)
p_w_true = expit(eta_w)
W_matrix = rng_w.binomial(1, p_w_true).astype(np.int64)
print("W shape:", W_matrix.shape, "| prevalence range:", W_matrix.mean(0).min(), W_matrix.mean(0).max())


# %% 2.5 Analysis arrays and development/held-out NCO split
X = X64.astype(np.float32)
Xt = torch.tensor(X, dtype=torch.float32)
At = torch.tensor(A.reshape(-1, 1), dtype=torch.float32)
ind_treated, ind_control = A == 1, A == 0

x_cols = [f"X{j+1}" for j in range(X_DIM)]
w_cols = [f"W{j+1:02d}" for j in range(N_NCO)]
df = pd.DataFrame(X, columns=x_cols)
df["treatment"] = A
df = pd.concat([df, pd.DataFrame(W_matrix, columns=w_cols)], axis=1)
             
column_names_list = list(df.columns)

NCO, W_all = [], []
for col in w_cols:
    i = int(df.columns.get_loc(col))
    W = df[col].to_numpy()
    if W.mean() >= MIN_NCO_PREVALENCE:
        NCO.append(i)
        # Also exclude NCOs that are constant at 1 in either treatment arm.
        if (MIN_EVENTS_PER_ARM <= W[ind_treated].mean() < 1
                and MIN_EVENTS_PER_ARM <= W[ind_control].mean() < 1):
            W_all.append(i)
if len(W_all) < 4:
    raise ValueError("Too few estimable NCOs for development / held-out split.")
W_split, W_test = train_test_split(
    W_all, test_size=0.5, random_state=NCO_SPLIT_SEED, shuffle=True,
)
assert set(W_split).isdisjoint(W_test)
print("Xt, At:", Xt.shape, At.shape)
print("Eligible / development / held-out NCOs:", len(W_all), len(W_split), len(W_test))
