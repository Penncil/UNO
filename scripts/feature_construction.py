from pathlib import Path

import numpy as np
import pandas as pd


def construct_features(
    data_path: str | Path,
    covariate_prefix: str,
    nco_prefix: str,
    treatment_column: str,
    outcome_column: str,
):
    """Load the analysis CSV, validate its schema, and construct model arrays."""
    df = pd.read_csv(data_path, on_bad_lines="error")
    covariate_columns = [c for c in df.columns if c.startswith(covariate_prefix)]
    nco_columns = [c for c in df.columns if c.startswith(nco_prefix)]

    missing = {treatment_column, outcome_column}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not covariate_columns:
        raise ValueError("No covariate columns were found.")
    if not nco_columns:
        raise ValueError("No NCO columns were found.")

    X = df[covariate_columns].to_numpy(dtype=np.float32)
    A = df[treatment_column].to_numpy(dtype=np.int64)
    Y = df[outcome_column].to_numpy(dtype=np.float64)
    W = df[nco_columns].to_numpy(dtype=np.int8)

    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Covariates and outcome cannot contain missing or infinite values.")
    if set(np.unique(A)) != {0, 1}:
        raise ValueError("Treatment must be binary and contain both groups.")
    if not set(np.unique(W)).issubset({0, 1}):
        raise ValueError("All NCO columns must be binary.")

    return df, covariate_columns, nco_columns, X, A, Y, W

