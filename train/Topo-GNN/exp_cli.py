import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Annotated

import typer

from fundus_vessels_toolkit.utils.nnet.experiment import ExperimentCfg, NoTrialsToRunError
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
        required_fields = [
            field_name
            for field_name, field_info in DigraphGNNTrainerConfig.model_fields.items()
            if field_info.is_required()
        ]
        if required_fields:
            for field_name in required_fields:
                f.write(f"{field_name}: <{field_name}>\n")


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


@app.command()
def single_run(
    file: Annotated[Path, typer.Argument(help="Path to the experiment configuration file to check.")],
):
    from train import train

    try:
        exp = ExperimentCfg.load_experiment(file, DigraphGNNTrainerConfig)
    except NoTrialsToRunError as e:
        raise typer.Exit(20) from None
    train(exp)


@app.command()
def sbatch(
    script: Annotated[Path, typer.Argument(help="Path to the bash script to submit.")],
    file: Annotated[Path, typer.Argument(help="Path to the experiment configuration file to run.")],
):
    print(f"Checking configuration file {file}...")
    if ExperimentCfg.check_file(file, DigraphGNNTrainerConfig):
        print("\t\t [OK]")
    else:
        print("Configuration file is not valid. Aborting.")
        return

    exp_header = ExperimentCfg.load_header(file)
    n_runs = exp_header.trials_to_run()
    if n_runs == 0:
        print("No remaining trials to run for this configuration. Aborting.")
        return

    job_uuid = uuid.uuid4().hex
    print(f"Submitting {n_runs} runs for experiment configuration {file} with UUID {job_uuid}...")

    # Move config file and bash script to dedicated folder
    job_dir = Path("tmp") / "JOBS" / job_uuid
    job_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(file, job_dir / "cfg.yaml")
    job_script = job_dir / "run.sh"

    # Replace field in bash script
    with open(script, "r") as script_file:
        script_txt = script_file.read()
    script_txt = script_txt.replace(r"{DIR}", str(job_dir.absolute()))
    script_txt = script_txt.replace(r"{EXP}", exp_header.experiment_name)
    script_txt = script_txt.replace(r"{N_RUNS}", str(n_runs))
    script_txt = script_txt.replace(r"{EXP_FILE}", str((job_dir / "cfg.yaml").absolute()))
    with open(job_script, "w") as script_file:
        script_file.write(script_txt)
    job_script.chmod(0o755)

    # Submit job
    submit_result = subprocess.run(["sbatch", str(job_script.absolute())], capture_output=True, text=True)
    if submit_result.returncode != 0:
        print(f"Failed to submit job: {submit_result.stderr}")
    else:
        print(f"Job submitted successfully: {submit_result.stdout}")


if __name__ == "__main__":
    app()
