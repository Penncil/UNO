# Fit Logistic PS and train the Base neural network.
# Run in order using run_simulation.py or simulation_benchmark.ipynb.

# %% 3. Logistic propensity-score baseline
LOGIT_C = 1.0
LOGIT_MAX_ITER = 2000
LOGIT_SEED = 2026

logit_ps_model = LogisticRegression(
    C=LOGIT_C, max_iter=LOGIT_MAX_ITER, solver="liblinear", random_state=LOGIT_SEED,
)
logit_ps_model.fit(X, A)
ps_logit = logit_ps_model.predict_proba(X)[:, 1]
print("Logistic treatment accuracy:", np.mean((ps_logit >= 0.5) == A))
print("Logistic PS range:", ps_logit.min(), ps_logit.max())


# %%  Base network settings and initialization
TRAIN_SEED = 2026
n_hidden = 100                 # UNO inherits the fitted Base architecture.
n_depth = 3
DROPOUT = 0.1
NN_PS_FLOOR = 0.01              # Network output: 0.01 + 0.98*sigmoid(logit).
n_epochs = 50
batch_size = 32
BASE_LR = 0.001
BASE_WEIGHT_DECAY = 5e-6
BASE_STEP_SIZE = 10
BASE_LR_GAMMA = 0.5
EARLY_STOP_ACCURACY = 0.95      # Set None to disable accuracy-based early stopping.

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


# %%   Base training (explicit loops)
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
base_rng_state = torch.get_rng_state().clone()  # Starting RNG state for pruning reruns.
BASE_CONFIG = {"in_N": d, "m": n_hidden, "depth": n_depth,
               "dropout": DROPOUT, "ps_floor": NN_PS_FLOOR}
BASE_MODEL_AT_TRAINING = copy.deepcopy(BASE_CONFIG)
print("Base training seconds:", round(time.perf_counter()-base_start, 2))
