import json
from pathlib import Path
from typing import Annotated

import typer

from fundus_vessels_toolkit.utils.nnet.experiment import ExperimentCfg
from fundus_vessels_toolkit.utils.nnet.pydantic_yaml import model_validate_yaml_file
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
        json.dump(ExperimentCfg.model_json_schema(), f)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(DigraphGNNTrainerConfig.model_json_schema(), f)

    template_path.parent.mkdir(parents=True, exist_ok=True)
    if template_path.exists():
        return
    experiment_path = experiment_path.relative_to(template_path.parent)
    config_path = config_path.relative_to(template_path.parent)
    with open(template_path, "w") as f:
        f.write(f"""
# yaml-language-server: $schema={experiment_path}
experiment: <experiment_name>
---
# yaml-language-server: $schema={config_path}
""")


@app.command()
def check(file: Annotated[Path, typer.Argument(help="Path to the experiment configuration file to check.")]):
    if ExperimentCfg.check_file(file, DigraphGNNTrainerConfig):
        print("Configuration file is valid.")


@app.command()
def test_run(
    file: Annotated[Path, typer.Argument(help="Path to the experiment configuration file to check.")],
    max_epoch: int = 25,
):
    from train import train

    exp = ExperimentCfg.load_experiment(
        file, DigraphGNNTrainerConfig, header_override={"test_debug": True}, override={"epoch": max_epoch}
    )
    train(exp)


if __name__ == "__main__":
    app()
