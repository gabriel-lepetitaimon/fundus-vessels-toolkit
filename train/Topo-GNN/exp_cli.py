import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from fundus_vessels_toolkit.utils.nnet.experiment import ExperimentCfg
from train import DigraphGNNTrainerConfig

app = typer.Typer()

EXP = Path("EXP")


@app.command()
def export_schema(
    experiment: str = typer.Argument(EXP / "experiment.schema.json", help="Path where to save the experiment schema."),
    config: str = typer.Argument(EXP / "config.schema.json", help="Path where to save the configuration schema."),
    template: str = typer.Argument(EXP / "exp_template.yaml", help="Path where to save the experiment template."),
):
    experiment_path = Path(experiment)
    config_path = Path(config)
    template_path = Path(template)

    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    with open(experiment_path, "w") as f:
        json.dump(ExperimentCfg.model_json_schema(), f, indent=4)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(DigraphGNNTrainerConfig.model_json_schema(), f, indent=4)

    template_path.parent.mkdir(parents=True, exist_ok=True)
    if template_path.exists():
        return
    experiment_path = experiment_path.relative_to(template_path.parent)
    config_path = config_path.relative_to(template_path.parent)
    with open(template_path, "w") as f:
        f.write(f"""
# yaml-language-server: $schema={experiment_path}
---
# yaml-language-server: $schema={config_path}
""")


@app.command()
def check(file: Annotated[Path, typer.Argument(help="Path to the experiment configuration file to check.")]):
    try:
        with open(file, "r") as f:
            config_dict = json.load(f)
        ExperimentCfg(**config_dict)
        print("Experiment configuration is valid.")
    except Exception as e:
        print(f"Experiment configuration is invalid: {e}")


if __name__ == "__main__":
    app()
