import os

os.environ["WANDB_MODE"] = "disabled"

from pathlib import Path

from exp_cli import test_run

test_run(Path("EXP/0a_hyper_param_search.yaml"), 1)
