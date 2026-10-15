"""Where HPC-DS1 finds its files, the same way from a git clone or a pip install.

Everything needed at run time ships inside the package:
    ui/        the facility console and layer explorer
    models/    the trained step model (with its familiarity guard) and System 1
    data/real/ the real measurements they were trained on
    results/   evaluation results shown on the console's Evaluation page

Per-machine state lives in ~/.hpc_ds1 ($HPC_DS1_HOME): measurements and models
from `hpc-ds1 setup`/`learn`, the audit log, live runs, and the question-encoder
weights (download from the GitHub release, about 600 MB; too big for PyPI).
"""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
UI_DIR = PACKAGE_DIR / "ui"
MODELS_DIR = PACKAGE_DIR / "models"
DATA_DIR = PACKAGE_DIR / "data" / "real"
RESULTS_DIR = PACKAGE_DIR / "results"


def home_dir():
    return Path(os.environ.get("HPC_DS1_HOME", Path.home() / ".hpc_ds1"))


def encoder_dir():
    """$HPC_DS1_ENCODER, else ~/.hpc_ds1/models/encoder, else models/encoder in a source checkout."""
    if os.environ.get("HPC_DS1_ENCODER"):
        return Path(os.environ["HPC_DS1_ENCODER"])
    own = home_dir() / "models" / "encoder"
    if (own / "scorer.pt").exists():
        return own
    return PACKAGE_DIR.parent / "models" / "encoder"
