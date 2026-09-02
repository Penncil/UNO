# UNO: Neural Pruning Enhances AI Reliability of Observational Data-based Findings

Clinical observational data are increasingly used in drug development to assess treatment effects across diverse populations. However, such analyses are vulnerable to systematic error from measured and unmeasured confounding, selection bias, and misclassification. AI pipelines offer modeling flexibility for high-dimensional data but may amplify spurious patterns in the presence of such biases. We propose the method of Unlearning Neural networks using Observational data (UNO), a principled machine unlearning framework that leverages negative control outcomes (NCOs) to mitigate systematic error in neural network-based models. UNO identifies bias-prone individuals using NCO estimates, prunes neurons most associated with the biased subset, and fine-tunes the model on the reliable subset. The approach is task-agnostic, architecture-agnostic, and improves model reliability without sacrificing expressiveness. We demonstrate its utility in two large-scale studies: profiling the clinical effects of GLP-1 receptor agonists versus DPP-4 inhibitors across 189 outcomes in more than 303,580 patients, and screening 33 candidate therapies for Alzheimer’s disease using longitudinal records from 16,184 patients. These results suggest a new paradigm of neural pruning to enhance the reliability of AI in drug development with observational data.


---
This repository contains a Jupyter notebook and explicit scripts for each stage of the UNO workflow. `UNO.ipynb` imports these scripts while retaining the step-by-step method explanation.

## Repository contents
```text
UNO/
├── README.md
├── UNO.ipynb
├── simulated_data.csv
├── scripts/
│   ├── feature_construction.py
│   ├── propensity_score_modeling.py
│   ├── neural_network_training.py
│   ├── neuron_scoring.py
│   ├── pruning.py
│   ├── fine_tuning.py
│   └── downstream_estimation.py
├── requirements.txt
└── LICENSE
```


## Installation

Python 3.11 is recommended.

```bash
git clone https://github.com/Penncil/UNO.git
cd UNO

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the demo

```bash
jupyter lab UNO.ipynb
```

Open the notebook and run its cells in order. The notebook contains eight clearly labeled sections:

1. package imports and analysis settings;
2. data loading and validation;
3. base propensity-score neural network training;
4. NCO modeling and individual-level signal grouping;
5. neuron activation scoring;
6. neuron pruning;
7. fine-tuning and downstream estimation;
8. saved results and run summary.

###   Expected output
- The notebook executes without errors
- Intermediate results (e.g., training progress, figures) are displayed inline
- Final outputs are generated within the notebook 

The demo uses simulated data and is intended only to verify that the code executes correctly and is consistent with the method. It does not reproduce the main results in the manuscript.

###   Expected run time for demo
- Approximately 2–5 minutes on a standard laptop for one outcome.






## System requirements

### Software requirements
- see requirements.txt

### Operating systems
The software has been tested on:
- macOS (Apple silicon / Intel)
The code is expected to be compatible with Linux-based systems.

### Hardware requirements
- CPU: Standard desktop computer
- GPU: Optional, not required

 

## Using another dataset

Change `DATA_PATH` and the column-name settings in the first notebook code cell. Treatment and NCO columns must be binary, and modeled columns must not contain missing or infinite values.

Cohort construction, eligibility criteria, temporal anchoring, clinical variable definitions, causal assumptions, balance diagnostics, and the uncertainty procedure remain study-specific responsibilities.

## License

MIT License. See `LICENSE`.

 
  

