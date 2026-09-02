import numpy as np
import torch
import torch.nn as nn


class PropensityNetwork(nn.Module):
    """Feed-forward neural network for propensity-score modeling."""

    def __init__(self, input_dim, hidden_width=300, depth=3, dropout=0.1):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one")

        self.hidden_layers = nn.ModuleList()
        for layer_index in range(depth):
            in_features = input_dim if layer_index == 0 else hidden_width
            self.hidden_layers.append(nn.Linear(in_features, hidden_width))

        self.output_layer = nn.Linear(hidden_width, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.dropout(self.activation(layer(x)))
        probability = torch.sigmoid(self.output_layer(x))
        return 0.01 + 0.98 * probability

    def hidden_activations(self, x):
        """Return post-ReLU, pre-dropout activations for UNO scoring."""
        activations = []
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            activations.append(x)
            x = self.dropout(x)
        return activations


def predict_propensity(model, x_array, device):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_array, dtype=torch.float32, device=device)
        return model(x_tensor).squeeze(1).cpu().numpy()

