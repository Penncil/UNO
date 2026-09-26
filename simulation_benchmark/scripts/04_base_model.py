# Initialize and train the original Base neural network.
# Source-preserving stage; execute with workflow.run_stage in the shared notebook namespace.
# Numerical expressions, parameter values, and statement order are unchanged.

# %% SOURCE_FRAGMENT 24; original zero-based cell 24; id da75590a
TRAIN_SEED = 2026
n_hidden = 100                 # Original width; UNO inherits the actual fitted Base architecture.
n_depth = 3
DROPOUT = 0.1
NN_PS_FLOOR = 0.01              # Original output: 0.01 + 0.98 * sigmoid(logit).
n_epochs = 50
batch_size = 32
BASE_LR = 0.001
BASE_WEIGHT_DECAY = 5e-6
BASE_STEP_SIZE = 10
BASE_LR_GAMMA = 0.5
EARLY_STOP_ACCURACY = 0.95      # Original stopping rule; None would disable accuracy-based stopping.

np.random.seed(TRAIN_SEED)
torch.manual_seed(TRAIN_SEED)
d = X.shape[1]
model = PS(in_N=d, m=n_hidden, depth=n_depth, dropout=DROPOUT, ps_floor=NN_PS_FLOOR)
model.apply(weights_init)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=BASE_WEIGHT_DECAY)
scheduler = StepLR(optimizer, step_size=BASE_STEP_SIZE, gamma=BASE_LR_GAMMA)
base_history = []
print(model)

# %% SOURCE_FRAGMENT 26; original zero-based cell 26; id a9b63beb
base_start = time.perf_counter()
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    permutation = torch.randperm(Xt.size(0))
    for i in range(0, Xt.shape[0], batch_size):
        ind = permutation[i:i+batch_size]
        batch_x, batch_a = Xt[ind], At[ind]
        out = model(batch_x)
        optimizer.zero_grad()
        loss = criterion(out, batch_a)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*len(ind)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        predicted_prob = model(Xt).numpy().reshape(-1)
    accuracy = float(np.mean((predicted_prob >= 0.5) == A))
    base_history.append({"epoch": epoch+1, "loss": running_loss/len(X),
                         "accuracy": accuracy, "lr": optimizer.param_groups[0]["lr"]})
    print(f"Base {epoch+1:02d}/{n_epochs}  loss={running_loss/len(X):.5f}  "
          f"accuracy={accuracy:.4f}  lr={optimizer.param_groups[0]['lr']:.2g}")
    if EARLY_STOP_ACCURACY is not None and accuracy > EARLY_STOP_ACCURACY:
        print("Stopped at the specified treatment-training accuracy threshold.")
        break

model.eval()
with torch.no_grad():
    ps_base = model(Xt).numpy().reshape(-1).copy()
trained = copy.deepcopy(model.state_dict())
base_rng_state = torch.get_rng_state().clone()  # Save the original RNG state used to initialize each pruning workflow.
BASE_CONFIG = {"in_N": d, "m": n_hidden, "depth": n_depth,
               "dropout": DROPOUT, "ps_floor": NN_PS_FLOOR}
BASE_MODEL_AT_TRAINING = copy.deepcopy(BASE_CONFIG)
print("Base training seconds:", round(time.perf_counter()-base_start, 2))
