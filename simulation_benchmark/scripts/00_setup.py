# Imports, fixed CPU settings, and numerical helpers.
# Run in order using run_simulation.py or simulation_benchmark.ipynb.

# %% 1. Imports and CPU settings
# Install missing packages in the active environment if needed.
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

# Use fixed CPU thread settings for the calculation stages.
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


# %% Import the unchanged basic definitions
from scripts.functions import (
    PS,
    M_pruned,
    weights_init,
    test_NCO,
    v1_risk_scores,
    ipw_weights,
    fit_null,
    compute_expected_absolute_systematic_error_null,
    RR,
    evaluate_nco_ease,
)
