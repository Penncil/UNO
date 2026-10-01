"""Run the complete UNO benchmark.

Usage: python run_simulation.py 
"""
from pathlib import Path
import sys

_RUN_ROOT = Path(__file__).resolve().parent
if str(_RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUN_ROOT))

_RUN_STAGES = [
    "00_setup.py",
    "01_generate_data.py",
    "02_train_base.py",
    "03_run_uno.py",
    "04_single_outcome.py",
    "05_repeated_outcomes.py",
    "06_nco_replacement.py",
    "07_save_results.py",
]

if __name__ == "__main__":
    for _run_stage_name in _RUN_STAGES:
        _run_stage_path = _RUN_ROOT / "scripts" / _run_stage_name
        print(f"\n--- {_run_stage_name} ---", flush=True)
        exec(
            compile(_run_stage_path.read_text(encoding="utf-8"), str(_run_stage_path), "exec"),
            globals(),
            globals(),
        )
