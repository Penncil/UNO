# Original NCO-sensitivity parameters (default: 100% replacement).
# Source-preserving stage; execute with workflow.run_stage in the shared notebook namespace.
# Numerical expressions, parameter values, and statement order are unchanged.

# %% SOURCE_FRAGMENT 51_parameters; original zero-based cell 51; id 879f32e5
# ---------- Original NCO-perturbation parameters ----------
NCO25_REPLACEMENT_FRACTION = 1 #0.1#0.75 #0.5 #0.25
NCO25_SELECTION_SEED = 2026    # Determines the replacement subset only; not selected using outcome results.
NCO25_W_SEED = 2026            # Independent replacement-NCO stream; does not advance the original rng_w.
