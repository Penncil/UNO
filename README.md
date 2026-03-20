# UNO: Neural Pruning Enhances AI Reliability of Observational Data-based Findings

Clinical observational data are increasingly used in drug development to assess treatment effects across diverse populations. However, such analyses are vulnerable to systematic error from measured and unmeasured confounding, selection bias, and misclassification. AI pipelines offer modeling flexibility for high-dimensional data but may amplify spurious patterns in the presence of such biases. We propose the method of Unlearning Neural networks using Observational data (UNO), a principled machine unlearning framework that leverages negative control outcomes (NCOs) to mitigate systematic error in neural network-based models. UNO identifies bias-prone individuals using NCO estimates, prunes neurons most associated with the biased subset, and fine-tunes the model on the reliable subset. The approach is task-agnostic, architecture-agnostic, and improves model reliability without sacrificing expressiveness. We demonstrate its utility in two large-scale studies: profiling the clinical effects of GLP-1 receptor agonists versus DPP-4 inhibitors across 189 outcomes in more than 303,580 patients, and screening 33 candidate therapies for Alzheimer’s disease using longitudinal records from 16,184 patients. These results suggest a new paradigm of neural pruning to enhance the reliability of AI in drug development with observational data.


---

This repository contains the code for the method described above.

## 1. System requirements

### 1.1 Software requirements
- Python: [3.11.0]
- Required packages:           
    - matplotlib==3.8.0                   
    - numpy==1.26.4                     
    - pandas==2.2.3     
    - pytorch==2.5.1            
    - scikit-learn==1.2.2            
    - scipy==1.11.4          
    - statsmodels==0.14.0

### 1.2 Operating systems
The software has been tested on:
- macOS (Apple silicon / Intel)
The code is expected to be compatible with Linux-based systems.

### 1.3 Hardware requirements
- CPU: Standard desktop computer
- GPU: Optional, not required

---

## 2. Installation guide

Clone the repository:

bash
git clone https://github.com/Penncil/UNO.git
cd UNO

Approximately 1–3 minutes on a standard desktop computer.

---

## 3. Demo

### 3.1 Instructions to Run
Launch Jupyter Notebook; Open UNO.ipynb; Run all cells.

### 3.2 Expected Output
- The notebook executes without errors
- Intermediate results (e.g., training progress, figures) are displayed inline
- Final outputs are generated within the notebook 

The demo uses simulated data and is intended only to verify that the code executes correctly and is consistent with the method. It does not reproduce the main results in the manuscript.

### 3.3 Expected run time for demo
- Approximately 2–5 minutes on a standard laptop for one outcome.

---

## 4. Instructions for Use

### 4.1 Running on Your Own Data
To apply the method to your own dataset, prepare your data in the following format:
- CSV file
- Columns:
  - covaraites
  - outcome
  - treatment variable
  - multiple negative control outcomes
- Modify the data loading section in the notebook: data = load_data("your_data.csv")
- Replace the column names in code by your data column names

### 4.2 Reproducibility (Optional)
Reproducing the main results in the manuscript requires access to the full dataset described in the paper.
The provided simulated data is intended for demonstration purposes only.

