# Fit the original logistic-regression propensity-score comparator.
# Source-preserving stage; execute with workflow.run_stage in the shared notebook namespace.
# Numerical expressions, parameter values, and statement order are unchanged.

# %% SOURCE_FRAGMENT 22; original zero-based cell 22; id 41fef89f
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
