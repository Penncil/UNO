import copy

import numpy as np
import pandas as pd
import torch


def make_pruning_masks(model, scores, pruning_ratio):
    if not 0 <= pruning_ratio < 1:
        raise ValueError("pruning_ratio must be between 0 and 1.")

    masks = [np.zeros(layer.out_features, dtype=bool) for layer in model.hidden_layers]
    number_to_prune = int(sum(len(mask) for mask in masks) * pruning_ratio)
    for row in scores.nlargest(number_to_prune, "score").itertuples(index=False):
        masks[int(row.layer)][int(row.neuron)] = True
    return masks


def enforce_pruning(model, masks, device):
    with torch.no_grad():
        for layer_index, mask_array in enumerate(masks):
            mask = torch.as_tensor(mask_array, dtype=torch.bool, device=device)
            current_layer = model.hidden_layers[layer_index]
            next_layer = (
                model.hidden_layers[layer_index + 1]
                if layer_index + 1 < len(model.hidden_layers)
                else model.output_layer
            )
            current_layer.weight[mask, :] = 0
            current_layer.bias[mask] = 0
            next_layer.weight[:, mask] = 0


def zero_pruned_gradients(model, masks, device):
    for layer_index, mask_array in enumerate(masks):
        mask = torch.as_tensor(mask_array, dtype=torch.bool, device=device)
        current_layer = model.hidden_layers[layer_index]
        next_layer = (
            model.hidden_layers[layer_index + 1]
            if layer_index + 1 < len(model.hidden_layers)
            else model.output_layer
        )
        current_layer.weight.grad[mask, :] = 0
        current_layer.bias.grad[mask] = 0
        next_layer.weight.grad[:, mask] = 0


def prune_model(base_model, neuron_scores, pruning_ratio, device):
    """Copy the base model, prune top-scoring neurons, and return its audit table."""
    model = copy.deepcopy(base_model)
    masks = make_pruning_masks(model, neuron_scores, pruning_ratio)
    enforce_pruning(model, masks, device)
    table = pd.DataFrame(
        [
            {"layer": layer_index, "neuron": neuron_index, "pruned": bool(is_pruned)}
            for layer_index, mask in enumerate(masks)
            for neuron_index, is_pruned in enumerate(mask)
        ]
    )
    return model, masks, table

