"""Execute source-preserving analysis stages in one shared notebook namespace.

The scientific code lives in scripts/. This loader performs no model fitting,
parameter substitution, random-number generation, or automatic reseeding.
Using a shared namespace preserves the original notebook's global variables and
function lookups. It also avoids changing the order of random draws through
module imports. Every call rereads the script, rather than importing cached code.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIMARY_STAGES = (
    "01_setup", "02_generate_data", "03_logistic_ps", "04_base_model",
    "05_nco_signals", "06_uno", "07_heldout_evaluation", "08_target_outcome",
    "09_empirical_calibration", "10_repeated_outcomes",
)
SENSITIVITY_STAGES = (
    "12_sensitivity_data", "13_sensitivity_development", "14_sensitivity_evaluation",
)
ALL_STAGES = PRIMARY_STAGES + ("11_sensitivity_parameters",) + SENSITIVITY_STAGES


def run_stage(name, namespace):
    """Run an original-order stage using the caller's global dictionary.

    Parameters
    ----------
    name : str
        A stage name from ALL_STAGES, without the .py suffix.
    namespace : dict
        Pass globals() from the runner notebook. Do not copy the dictionary:
        the original functions must resolve variables in the same namespace.
    """
    if name not in ALL_STAGES:
        raise ValueError(f"Unknown analysis stage: {name!r}")
    if not isinstance(namespace, dict):
        raise TypeError("namespace must be the notebook's globals() dictionary")
    path = ROOT / "scripts" / f"{name}.py"
    source = path.read_text(encoding="utf-8")
    # Do not inherit compile-time flags from this wrapper. No code is rewritten.
    code = compile(source, str(path), "exec", dont_inherit=True, optimize=0)
    exec(code, namespace, namespace)
