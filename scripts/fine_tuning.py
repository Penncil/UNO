import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from scripts.propensity_score_modeling import predict_propensity
from scripts.pruning import enforce_pruning, zero_pruned_gradients


def fine_tune_pruned_model(
    model,
    x_array,
    treatment,
    reliable_mask,
    masks,
    device,
    epochs=3,
    learning_rate=1e-4,
    weight_decay=5e-6,
    batch_size=80,
):
    """Fine-tune the pruned model on low-NCO-signal individuals."""
    x_reliable = torch.as_tensor(x_array[reliable_mask], dtype=torch.float32, device=device)
    a_reliable = torch.as_tensor(treatment[reliable_mask, None], dtype=torch.float32, device=device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    history = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(x_reliable), device=device)
        total_loss = 0.0

        for start in range(0, len(x_reliable), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_reliable[indices]), a_reliable[indices])
            loss.backward()
            zero_pruned_gradients(model, masks, device)
            optimizer.step()
            enforce_pruning(model, masks, device)
            total_loss += loss.item() * len(indices)

        scheduler.step()
        full_probability = predict_propensity(model, x_array, device)
        full_accuracy = float(((full_probability >= 0.5) == treatment).mean())
        row = {
            "epoch": epoch + 1,
            "reliable_subset_loss": total_loss / len(x_reliable),
            "full_population_accuracy": full_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row)

    return pd.DataFrame(history)

