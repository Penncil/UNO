import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _arm_specific_nco_probability(x_array, outcome, arm_mask, seed):
    arm_outcome = outcome[arm_mask]
    if np.unique(arm_outcome).size == 1:
        return np.full(len(x_array), float(arm_outcome[0]))

    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)
    model.fit(x_array[arm_mask], arm_outcome)
    return model.predict_proba(x_array)[:, 1]


def split_negative_controls(
    nco_values,
    treatment,
    split_seed=2026,
    minimum_prevalence=0.001,
):
    """Split eligible NCOs into signal, ratio-tuning, and final-test sets."""
    eligible_indices = []
    for nco_index in range(nco_values.shape[1]):
        outcome = nco_values[:, nco_index]
        if outcome[treatment == 1].mean() < minimum_prevalence:
            continue
        if outcome[treatment == 0].mean() < minimum_prevalence:
            continue
        eligible_indices.append(nco_index)

    if len(eligible_indices) < 6:
        raise ValueError("At least six eligible NCOs are required for the 1/3, 1/6, 1/2 split.")

    W_dev, W_test = train_test_split(
        eligible_indices,
        test_size=0.5,
        random_state=split_seed,
        shuffle=True,
    )
    W_split, W_ratio = train_test_split(
        W_dev,
        test_size=1 / 3,
        random_state=split_seed + 1,
        shuffle=True,
    )

    W_split = np.asarray(W_split, dtype=int)
    W_ratio = np.asarray(W_ratio, dtype=int)
    W_test = np.asarray(W_test, dtype=int)
    eligible_indices = np.asarray(eligible_indices, dtype=int)

    assert set(W_split).isdisjoint(W_ratio)
    assert set(W_split).isdisjoint(W_test)
    assert set(W_ratio).isdisjoint(W_test)
    assert set(W_split) | set(W_ratio) | set(W_test) == set(eligible_indices)
    return eligible_indices, W_split, W_ratio, W_test


def identify_nco_signal(
    x_array,
    treatment,
    nco_values,
    nco_columns,
    selected_indices,
    seed=25,
    risk_difference_threshold=0.05,
    consensus_fraction=0.50,
):
    """Fit treatment-stratified NCO models and define high/low-NCO-signal subsets."""
    nco_prevalence = nco_values.mean(axis=0)
    selected_indices = np.asarray(selected_indices, dtype=int)
    if len(selected_indices) == 0:
        raise ValueError("The signal-grouping NCO set is empty.")

    signal_flags = []
    diagnostics = []
    for nco_index in selected_indices:
        nco_outcome = nco_values[:, nco_index]
        predicted_if_untreated = _arm_specific_nco_probability(
            x_array, nco_outcome, treatment == 0, seed
        )
        predicted_if_treated = _arm_specific_nco_probability(
            x_array, nco_outcome, treatment == 1, seed
        )
        absolute_difference = np.abs(predicted_if_treated - predicted_if_untreated)
        signal = absolute_difference > risk_difference_threshold
        signal_flags.append(signal)
        diagnostics.append(
            {
                "nco": nco_columns[nco_index],
                "prevalence": nco_prevalence[nco_index],
                "mean_absolute_risk_difference": absolute_difference.mean(),
                "fraction_over_threshold": signal.mean(),
            }
        )

    signal_matrix = np.column_stack(signal_flags)
    required_signals = max(1, int(np.ceil(len(selected_indices) * consensus_fraction)))
    high_mask = signal_matrix.sum(axis=1) >= required_signals
    low_mask = ~high_mask
    if high_mask.all() or low_mask.all():
        raise ValueError("NCO settings produced an empty high- or low-signal subset.")

    return high_mask, low_mask, pd.DataFrame(diagnostics)


def score_all_neurons(
    model,
    x_array,
    high_mask,
    device,
    epsilon=1e-12,
    activation_batch_size=4096,
):
    """Calculate the UNO mean-squared activation ratio for every hidden neuron."""
    model.eval()
    high_sum = None
    low_sum = None
    high_count = int(high_mask.sum())
    low_count = int((~high_mask).sum())

    with torch.no_grad():
        for start in range(0, len(x_array), activation_batch_size):
            stop = min(start + activation_batch_size, len(x_array))
            x_batch = torch.as_tensor(x_array[start:stop], dtype=torch.float32, device=device)
            activations = model.hidden_activations(x_batch)
            batch_high = high_mask[start:stop]

            if high_sum is None:
                high_sum = [np.zeros(a.shape[1], dtype=np.float64) for a in activations]
                low_sum = [np.zeros(a.shape[1], dtype=np.float64) for a in activations]

            for layer_index, activation in enumerate(activations):
                values = activation.cpu().numpy().astype(np.float64)
                high_sum[layer_index] += np.square(values[batch_high]).sum(axis=0)
                low_sum[layer_index] += np.square(values[~batch_high]).sum(axis=0)

    rows = []
    for layer_index, (layer_high, layer_low) in enumerate(zip(high_sum, low_sum)):
        mean_high = layer_high / high_count
        mean_low = layer_low / low_count
        scores = mean_high / (mean_low + epsilon)
        for neuron_index, score in enumerate(scores):
            rows.append(
                {
                    "layer": layer_index,
                    "neuron": neuron_index,
                    "mean_square_high": mean_high[neuron_index],
                    "mean_square_low": mean_low[neuron_index],
                    "score": score,
                }
            )

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
