import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from scripts.propensity_score_modeling import PropensityNetwork, predict_propensity


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)


def train_base_model(
    x_array,
    treatment,
    device,
    hidden_width=300,
    depth=3,
    dropout=0.1,
    epochs=25,
    learning_rate=1e-3,
    weight_decay=5e-6,
    batch_size=80,
    early_stop_accuracy=0.95,
):
    """Train the base propensity-score neural network."""
    model = PropensityNetwork(
        input_dim=x_array.shape[1],
        hidden_width=hidden_width,
        depth=depth,
        dropout=dropout,
    ).to(device)
    model.apply(initialize_weights)

    x_tensor = torch.as_tensor(x_array, dtype=torch.float32, device=device)
    a_tensor = torch.as_tensor(treatment[:, None], dtype=torch.float32, device=device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    history = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(x_tensor), device=device)
        total_loss = 0.0

        for start in range(0, len(x_tensor), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_tensor[indices]), a_tensor[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)

        scheduler.step()
        probability = predict_propensity(model, x_array, device)
        accuracy = float(((probability >= 0.5) == treatment).mean())
        row = {
            "epoch": epoch + 1,
            "loss": total_loss / len(x_array),
            "accuracy": accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row)

        if accuracy > early_stop_accuracy:
            break

    return model, pd.DataFrame(history)

