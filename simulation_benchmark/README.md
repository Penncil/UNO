# Known-causal-effect simulation benchmark

This folder contains the fully simulated treatment-effect benchmark and the
sensitivity analysis that removes direct unmeasured-variable dependence from
selected development negative control outcomes (NCOs). It is separate from the
repository's original demonstration notebook and does not require clinical data
or `simulated_data.csv` from the repository root.


## Files

```text
simulation_benchmark/
    ├── run_simulation.py
    ├── simulation_benchmark.ipynb
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    └── scripts        
```

| File | Contents |
|---|---|
| `scripts/functions.py` | PS networks, post-ReLU activations, initialization, risk scores, IPW, Poisson RR, Gaussian-null fitting, and EASE |
| `scripts/00_setup.py` | Imports and fixed CPU/thread settings |
| `scripts/01_generate_data.py` | Data parameters; generate X, U, A, and W; fix the development/held-out NCO split |
| `scripts/02_train_base.py` | Logistic PS and explicit Base training loops |
| `scripts/03_run_uno.py` | NCO-CATE signals, high/low grouping, explicit pruning/fine-tuning loops, and held-out EASE |
| `scripts/04_single_outcome.py` | Generate the current Y; single-outcome RR, confidence intervals, and NCO-only empirical calibration |
| `scripts/05_repeated_outcomes.py` | 1000 paired Y realizations, RR errors, EC results, and paired comparisons |
| `scripts/06_nco_replacement.py` | The original development-NCO replacement experiment |
| `scripts/07_save_results.py` | Save current tables, arrays, model weights, and software versions |


## Run

Keep the directory structure intact. From the project directory:

```bash
python -m pip install -r requirements.txt
python run_simulation.py
```

Alternatively, open `simulation_benchmark.ipynb` and select **Restart Kernel and
Run All**. Each notebook cell runs the corresponding script with `%run -i`, so
all stages share the same kernel state. The notebook and command-line runner
execute the same script files, rather than maintaining duplicate copies of the
calculation code.

Run the stages in order.

  
## Edit parameters

Edit the relevant script, then rerun it and the downstream stages in the notebook.
The same edit is automatically used by `python run_simulation.py`.

- Data parameters: `01_generate_data.py`.
- Logistic/Base parameters: `02_train_base.py`.
- Grouping, pruning and fine-tuning parameters: `03_run_uno.py`.
- Outcome function, RR trimming and EC settings: `04_single_outcome.py`.
- Number of outcome-only repetitions: `05_repeated_outcomes.py`.
- Replacement fraction and random seeds: `06_nco_replacement.py`.
- Output directory: `07_save_results.py`.
 


## Outputs

The final stage writes to `results/`, relative to the working directory. Matching
filenames are overwritten on rerun. The folder is excluded by `.gitignore`.

Important outputs include:

- `ease_summary.csv`, `heldout_nco_estimates.csv`.
- `single_rr_summary_with_ec.csv`: single-Y RR estimates and confidence limits.
- `comparison_with_ec.csv`, `rr_replicates_with_ec.csv`: repeated-outcome results.
- `ec_calibration_parameters.csv`, `paired_comparisons.csv`.
- `nco_replacement_summary.csv`, `nco_replacement_rr_replicates.csv`.
- `nco_fraction_summary.csv`: replacement scenarios actually completed this session.
- `simulation_arrays.npz`, `base_model.pt`, `uno_model.pt`,
  `nco_replacement_uno_model.pt`, and `software_versions.json`.

Load arrays with `np.load("results/simulation_arrays.npz", allow_pickle=False)`.
No saved outputs or model files from an earlier session are bundled with the code.
The released notebook has empty outputs and execution counts.


## Interpretation
 
NCO-only empirical calibration shifts target log-RR estimates using a
Gaussian systematic-error mean estimated from development NCOs. It does not alter
propensity scores or weights. The calibration rows' held-out EASE is the parent
model's EASE. The original interval calculations use the
original model-based standard errors and fixed calibration parameters.

The sensitivity analysis regenerates selected development NCOs after removing
only the direct U term, leaving their observed-covariate dependence intact. 
The target-outcome errors summarize **one cohort and fixed fitted models**, with
outcome-only repetitions, not independent cohorts or repeated model fitting.
 
 

 
