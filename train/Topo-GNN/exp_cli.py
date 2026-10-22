import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Annotated

import typer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from fundus_vessels_toolkit.utils.nnet.experiment import ExperimentHeader, NoTrialsToRunError

app = typer.Typer(pretty_exceptions_show_locals=False)

EXP = Path("EXP")


@app.command()
def export_schema(
    experiment: str = typer.Argument(EXP / "experiment.schema.json", help="Path where to save the experiment schema."),
    config: str = typer.Argument(EXP / "config.schema.json", help="Path where to save the configuration schema."),
    template: str = typer.Argument(EXP / "exp_template.yaml", help="Path where to save the experiment template."),
):
    from train import DigraphGNNTrainerConfig

    experiment_path = Path(experiment)
    config_path = Path(config)
    template_path = Path(template)

    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    with open(experiment_path, "w") as f:
        json.dump(ExperimentHeader.model_json_schema(), f)

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
""")
        for field_name, field_info in ExperimentHeader.model_fields.items():
            if field_info.is_required():
                f.write(f"{field_name}: <{field_name}>\n")
        f.write(f"""
---
# yaml-language-server: $schema={config_path}
""")
        for field_name, field_info in DigraphGNNTrainerConfig.model_fields.items():
            if field_info.is_required():
                f.write(f"{field_name}: <{field_name}>\n")


@app.command()
def check(file: Annotated[Path, typer.Argument(help="Path to the experiment configuration file to check.")]):
    from train import DigraphGNNTrainerConfig

    if ExperimentHeader.check_file(file, DigraphGNNTrainerConfig):
        print("Configuration file is valid.")


@app.command()
def test_run(
    file: Annotated[Path, typer.Argument(help="Path to the experiment configuration file to check.")],
    max_epoch: int = 25,
    config: int = 0,
):
    from train import DigraphGNNTrainerConfig, train

    exp = ExperimentHeader.load_experiment(
        file,
        DigraphGNNTrainerConfig,
        header_override={"test_debug": True, "progress_bar": True},
        override={"epoch": max_epoch},
    )
    train(exp)


@app.command()
def single_run(
    file: Annotated[Path, typer.Argument(help="Path to the experiment configuration file to check.")],
):
    from train import DigraphGNNTrainerConfig, train

    try:
        exp = ExperimentHeader.load_experiment(file, DigraphGNNTrainerConfig)
    except NoTrialsToRunError as e:
        raise typer.Exit(20) from None
    train(exp)


@app.command()
def run_all(
    file: Annotated[Path, typer.Argument(help="Path to the experiment configuration file to check.")],
):
    retry_count = 0
    while True:
        out = subprocess.run([sys.executable, __file__, "single-run", file], capture_output=False)
        if out.returncode == 20:
            print("No more trials to run. Exiting.")
            break
        elif out.returncode not in [0, 21]:
            print(f"Error during run: {out.stderr}")
            retry_count += 1
            if retry_count > 3:
                print("Too many retries. Exiting.")
                break
            print("Retrying...")
        else:
            retry_count = 0


@app.command()
def sbatch(
    script: Annotated[Path, typer.Argument(help="Path to the bash script to submit.")],
    file: Annotated[Path, typer.Argument(help="Path to the experiment configuration file to run.")],
    no_check: Annotated[bool, typer.Option(help="Skip checking the configuration file.")] = False,
):
    if not no_check:
        from train import DigraphGNNTrainerConfig

        print(f"Checking configuration file {file}...")
        if ExperimentHeader.check_file(file, DigraphGNNTrainerConfig):
            print("\t\t [OK]")
        else:
            print("Configuration file is not valid. Aborting.")
            return

    exp_header = ExperimentHeader.load_header(file)
    exp_name = exp_header.experiment_name
    n_runs = exp_header.trials_to_run(split_by_parameters=True)
    if sum(n_runs) == 0:
        print("No remaining trials to run for experiment {exp_name}. Aborting.")
        return

    # Replace field in bash script
    with open(script, "r") as script_file:
        script_txt = script_file.read()

    if "MAX_CONCURRENT_RUNS" not in script_txt:
        submit_job(script_txt, file, exp_name, sum(n_runs))
    else:
        param_grid_names = list(exp_header.parameters_grid.keys())
        for param_grid_id, n_run in enumerate(n_runs):
            print("-" * 80)
            if n_run > 0:
                job_name = f"{exp_name}_{param_grid_names[param_grid_id]}"
                if not submit_job(script_txt, file, job_name, int(n_run), param_grid_id, 3):
                    return


def submit_job(
    script_txt: str, file: Path, job_name: str, n_runs: int, param_grid_id: int | None = None, max_concurrent_runs=1
) -> bool:
    job_uuid = uuid.uuid4().hex
    print(f"Submitting {n_runs} runs for experiment {job_name} with UUID {job_uuid}...")

    # Move config file and bash script to dedicated folder
    job_dir = Path("tmp") / "JOBS" / job_uuid
    job_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(file, job_dir / "cfg.yaml")
    job_script = job_dir / "run.sh"

    script_txt = script_txt.replace(r"{DIR}", str(job_dir.absolute()))
    script_txt = script_txt.replace(r"{EXP}", job_name)
    script_txt = script_txt.replace(r"{N_RUNS}", str(n_runs))
    script_txt = script_txt.replace(r"{MAX_CONCURRENT_RUNS}", str(max_concurrent_runs))

    EXP_FILE = str((job_dir / "cfg.yaml").absolute())
    if param_grid_id is not None:
        EXP_FILE += f":{param_grid_id}"
    script_txt = script_txt.replace(r"{EXP_FILE}", EXP_FILE)

    with open(job_script, "w") as script_file:
        script_file.write(script_txt)
    job_script.chmod(0o755)

    # Submit job
    submit_result = subprocess.run(["sbatch", str(job_script.absolute())], capture_output=True, text=True)
    if submit_result.returncode != 0:
        print(f"   Failed to submit job: {submit_result.stderr}")
        shutil.rmtree(job_dir)
        return False
    else:
        print(f"   Job submitted successfully: {submit_result.stdout}")
        return True


if __name__ == "__main__":
    app()
