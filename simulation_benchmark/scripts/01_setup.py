# Imports, deterministic CPU settings, model definitions, and estimators.
# Source-preserving stage; execute with workflow.run_stage in the shared notebook namespace.
# Numerical expressions, parameter values, and statement order are unchanged.

# %% SOURCE_FRAGMENT 2; original zero-based cell 2; id 5981dd58
# Install dependencies from requirements.txt before running this workflow.
# %pip install numpy pandas scipy torch scikit-learn statsmodels threadpoolctl

import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import norm
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels
import statsmodels.api as smapi
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from threadpoolctl import threadpool_limits
from IPython.display import display

# Preserve the original single-threaded CPU and deterministic-algorithm settings.
threadpool_limits(limits=1)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

versions = {
    "python": platform.python_version(), "numpy": np.__version__,
    "pandas": pd.__version__, "scipy": scipy.__version__,
    "torch": torch.__version__, "sklearn": sklearn.__version__,
    "statsmodels": statsmodels.__version__,
}
print(versions)

# %% SOURCE_FRAGMENT 4; original zero-based cell 4; id d71ba0f8
class PS(nn.Module):
    def __init__(self, in_N, m, depth=3, dropout=0.1, ps_floor=0.01):
        super().__init__()
        if min(in_N, m, depth) < 1 or not 0 <= ps_floor < 0.5:
            raise ValueError("Invalid network dimensions or PS output floor.")
        self.stack = nn.ModuleList([nn.Linear(in_N, m)])
        for _ in range(depth - 1):
            self.stack.append(nn.Linear(m, m))
        self.stack.append(nn.Linear(m, 1))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.ps_floor = float(ps_floor)

    def forward(self, x):
        for layer in self.stack[:-1]:
            x = self.dropout(self.act(layer(x)))
        return self.ps_floor + (1 - 2*self.ps_floor)*self.sigmoid(self.stack[-1](x))


class M_pruned(PS):
    def pre_act(self, x ): 
        p_list = []
        for layer in self.stack[:-1]:
            xi = layer(x)
            activated = self.act(xi)
            p_list.append(  activated)
            x = self.dropout(activated)
        ps = self.ps_floor + (1 - 2*self.ps_floor)*self.sigmoid(self.stack[-1](x))
        return ps, p_list


def weights_init(layer):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.0)


def test_NCO(nco_model, X, W=None):
    # W is retained for call compatibility; predictions do not use W.
    return nco_model.predict_proba(X)[:, 1]

# %% SOURCE_FRAGMENT 6; original zero-based cell 6; id 5b41c053
def v1_risk_scores(X, U, rho, c_coefficients, s_coefficients, v_coefficients,
                   tail_weights, u_weights):
    """Archived V1 scores. Input X: 20 columns; hidden U: 5 columns."""
    X = np.asarray(X, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    c = np.asarray(c_coefficients, dtype=np.float64)
    s = np.asarray(s_coefficients, dtype=np.float64)
    v = np.asarray(v_coefficients, dtype=np.float64)
    b = np.asarray(tail_weights, dtype=np.float64).copy()
    a = np.asarray(u_weights, dtype=np.float64)
    if (X.ndim != 2 or U.ndim != 2 or X.shape[1] != 20 or U.shape[1] != 5
            or X.shape[0] != U.shape[0]):
        raise ValueError("V1 requires matched X with 20 columns and U with 5 columns.")
    if (c.shape != (4,) or s.shape != (5,) or v.shape != (4,)
            or b.shape != (7,) or a.shape != (5,)):
        raise ValueError("Check the V1 coefficient and weight dimensions.")
    if not all(np.isfinite(z).all() for z in (X, U, c, s, v, b, a)):
        raise ValueError("Nonfinite V1 score inputs.")
    if not abs(rho) < 1 or np.linalg.norm(b) == 0:
        raise ValueError("Need abs(rho)<1 and nonzero tail weights.")
    b /= np.linalg.norm(b)
    Sigma_UU = rho ** np.abs(np.arange(5)[:, None] - np.arange(5)[None, :])
    u_sd = np.sqrt(a @ Sigma_UU @ a)
    if not np.isfinite(u_sd) or u_sd <= 0:
        raise ValueError("The hidden linear combination must have positive variance.")

    C = (c[0] * np.sin(X[:, 1])
         + c[1] * np.tanh(X[:, 2] * X[:, 3] - rho)
         + c[2] * np.tanh(X[:, 4] ** 2 - 1.0)
         + c[3] * np.tanh(X[:, 5] + X[:, 6]))
    S = (s[0] * np.tanh(X[:, 7])
         + s[1] * np.tanh(X[:, 8] * X[:, 9] - rho)
         + s[2] * np.tanh(X[:, 10] ** 2 - 1.0)
         + s[3] * np.sin(X[:, 11] + X[:, 12])
         + s[4] * np.tanh(X[:, 13:20] @ b))
    V = (v[0] * (U @ a) / u_sd
         + v[1] * np.tanh(U[:, 0] * U[:, 1] - rho)
         + v[2] * np.tanh(U[:, 2] ** 2 - 1.0)
         + v[3] * np.sin(U[:, 3] + U[:, 4]))
    return C, S, V

# %% SOURCE_FRAGMENT 8; original zero-based cell 8; id 8d7c4462
def ipw_weights(A, ps=None, eps=None):
    A = np.asarray(A).reshape(-1)
    if not np.isin(A, [0, 1]).all() or np.unique(A).size != 2:
        raise ValueError("A must be binary with both arms present.")
    if ps is None:
        return np.ones(A.size, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64).reshape(-1)
    if ps.shape != A.shape or not np.isfinite(ps).all():
        raise ValueError("PS is nonfinite or has the wrong shape.")
    if eps is not None:
        if not 0 < eps < 0.5:
            raise ValueError("PS_EPS must be None or in (0, 0.5).")
        ps = np.clip(ps, eps, 1-eps)
    if np.any((ps <= 0) | (ps >= 1)):
        raise ValueError("PS reached 0/1. Set a shared PS_EPS explicitly if intended.")
    return np.where(A == 1, 1/ps, 1/(1-ps))


def fit_null(logRr, seLogRr):
    y = np.asarray(logRr, dtype=float)
    se = np.asarray(seLogRr, dtype=float)
    if y.ndim != 1 or y.shape != se.shape or y.size == 0:
        raise ValueError("Effects and SEs must be nonempty vectors of equal size.")
    if not (np.isfinite(y).all() and np.isfinite(se).all() and (se > 0).all()):
        raise ValueError("Invalid NCO effects/SEs; no NCO is silently dropped.")

    def objective(theta):
        mu, log_sd = theta
        return float(-np.sum(norm.logpdf(
            y, loc=mu, scale=np.sqrt(se**2 + np.exp(log_sd)**2)
        )))

    result = minimize(
        objective, [float(y.mean()), np.log(max(float(y.std()), 1e-6))],
        method="L-BFGS-B", bounds=[(None, None), (np.log(1e-6), None)],
    )
    if not result.success:
        raise RuntimeError(f"Gaussian-null fit failed: {result.message}")
    return {"mean": float(result.x[0]), "sd": float(np.exp(result.x[1])),
            "n_nco": len(y), "negative_log_likelihood": float(result.fun)}


def compute_expected_absolute_systematic_error_null(null):
    mu, sd = null["mean"], null["sd"]
    if sd <= 1e-12:
        return abs(mu)
    z = mu / sd
    return float(sd*np.sqrt(2/np.pi)*np.exp(-0.5*z*z) + mu*(2*norm.cdf(z)-1))


def RR(weights, A, Y):
    # Preserve RR(weights, A, Y); return the log-RR and model-based standard error.
    A = np.asarray(A).reshape(-1)
    Y = np.asarray(Y).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if not (A.shape == Y.shape == weights.shape):
        raise ValueError("A, Y and weights must have the same shape.")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Weights must be finite and positive.")
    if not np.isin(Y, [0, 1]).all() or np.unique(A).size != 2:
        raise ValueError("Binary Y and both treatment arms are required.")
    if any(Y[A == a].sum() == 0 for a in (0, 1)):
        raise ValueError("An arm has zero events; not silently dropping this outcome.")
    AA = smapi.add_constant(pd.Series(A, name="A"), has_constant="add")
    fit = smapi.GLM(Y, AA, family=smapi.families.Poisson(),
                    freq_weights=weights).fit()
    if not fit.converged:
        raise RuntimeError("Poisson GLM did not converge.")
    return float(fit.params["A"]), float(fit.bse["A"])


def evaluate_nco_ease(df, column_names_list, nco_indices, A, weights):
    # Evaluation helper only; training and fine-tuning remain explicit loops.
    rows = []
    for i in nco_indices:
        W = df[column_names_list[i]].to_numpy()
        logrr, se = RR(weights, A, W)
        rows.append({"nco_index": int(i), "nco": column_names_list[i],
                     "log_rr": logrr, "se_log_rr": se})
    table = pd.DataFrame(rows)
    null = fit_null(table["log_rr"].to_numpy(), table["se_log_rr"].to_numpy())
    ease = compute_expected_absolute_systematic_error_null(null)
    return ease, table, null
