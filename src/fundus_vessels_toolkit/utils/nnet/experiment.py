from __future__ import annotations

import itertools
import tempfile
from contextvars import ContextVar, Token
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Sequence, Union, overload

import numpy as np
import optuna
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    StringConstraints,
    ValidationError,
    computed_field,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from rich.console import Console
from ruamel.yaml import YAML

import wandb
from fundus_toolkits.utils.typing import Int1DArray

from .optuna import OptunaCfg, TrialContext, current_trial
from .pydantic_yaml import InvalidDocumentCountError, YamlDocument, YamlDocumentWithFile, pretty_validation_error_msg


class NoTrialsToRunError(RuntimeError):
    """Raised when trying to load an experiment that has no remaining trials to run."""

    def __init__(self, header: ExperimentHeader, file: str | Path):
        super().__init__(
            f"All trials for this experiment have already been completed. No remaining trials to run for experiment defined in {file}."  # noqa: E501
        )


ParameterGridJSONInputType = Union[
    list[
        Annotated[
            dict[Annotated[str, StringConstraints(pattern="^[a-zA-Z_][a-zA-Z0-9_]*$")], dict[str, Any]],
            Field(min_length=1, max_length=1),
        ]
    ],
    dict[str, list],
]


def _validate_parameters_grid(data: ParameterGridJSONInputType) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]]
    if isinstance(data, dict):
        keys, values = list(data.keys()), list(data.values())
        values = [dict(zip(keys, v, strict=True)) for v in itertools.product(*values)]
        out = {"|".join(f"{k}={v[k]}" for k in keys): v for v in values}
        if len(out) == 1:
            out = {"": next(iter(out.values()))}
        return out
    elif isinstance(data, list):
        if len(data) == 0:
            return {}
        out = {}
        keys0: None | set[str] = None
        for d in data:
            if len(d) != 1:
                raise ValueError("Each dictionary in the parameters_grid list must have exactly one key.")
            name, values = next(iter(d.items()))
            if not isinstance(values, dict):
                raise ValueError(
                    "The value of each dictionary in the parameters_grid list must be another dictionary mapping parameter names to their values."  # noqa: E501
                )
            if keys0 is None:
                keys0 = set(values.keys())
            elif set(values.keys()) != keys0:
                raise ValueError("All dictionaries in the parameters_grid list must have the same keys.")
            out[name] = values
        return out
    else:
        raise ValueError("parameters_grid must be either a dict of lists or a list of key-dicts pairs.")


ParameterGridField = Annotated[
    dict[str, dict[str, Any]],
    BeforeValidator(_validate_parameters_grid, json_schema_input_type=ParameterGridJSONInputType),
]


class ExperimentHeader(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True, frozen=True)

    project: str
    """Name of the project this experiment belongs to."""

    experiment: str
    """Name of the experiment. This is used for logging and can be used to group related experiments together."""

    topic: Optional[str] = Field(default=None)
    """Topic or category of the experiment. This can be used for further organizing experiments."""

    version: Optional[int] = Field(default=None)
    """Version of the experiment. This can be used for tracking different iterations of the same experiment."""

    tags: list[str] = Field(default_factory=list)
    """List of tags to associate with this experiment. This can be used for filtering and organizing experiments."""

    n_trials: int = Field(default=20)
    """Number of trials to run. Default is 20."""

    log_model: bool | Literal["all"] = Field(default=True)

    test_debug: bool = Field(default=False)
    """Whether this is a test/debug run. If True, the experiment will use a temporary Optuna storage and will log to Wandb Test & Debug project, to avoid polluting the experiment results with test runs."""  # noqa: E501

    parameters_grid: ParameterGridField = Field(default_factory=dict)
    """Grid of parameters values to explore. This experiment will be run for each combination of parameters.
    The value of the parameters defined here can be referred to in the configuration file as '$<parameter_name>'
    
    This parameter can be defined in two formats:
    1. As a dictionary of lists, where each key is a parameter name and the value is a list of values to try for that parameter. For example:
    ```
    parameters_grid:
      learning_rate: [0.001, 0.01]
      batch_size: [32, 64]
    ```

    2. As a list pairing a name to a dictionary of parameter values. This allows for more flexibility in defining the parameter combinations, as the parameter names can be different for each combination. For example:
    ```
    parameters_grid:
      - experiment1:
          learning_rate: 0.001
          batch_size: 32
      - experiment2:
          learning_rate: 0.01
          batch_size: 64
    ```

    """  # noqa: E501

    optuna: OptunaCfg = Field(default_factory=OptunaCfg)
    """Optuna configuration for hyperparameter optimization."""

    progress_bar: bool = Field(default=False)
    """Whether to show a progress bar during training. This can be set to False to reduce console output when running many trials."""  # noqa: E501

    verbose: bool = Field(default=True)
    """Whether to show verbose output during training. """

    _file: Optional[Path] = PrivateAttr(default=None)

    @property
    def total_trials_count(self) -> int:
        """Total number of trials to run for this experiment, calculated as the product of the number of parameter combinations and the number of Optuna trials."""  # noqa: E501
        return len(self.parameters_grid) * self.n_trials

    @overload
    def current_trials_count(
        self, *, split_by_parameters: Literal[False] = False, only_completed: bool = False
    ) -> int: ...
    @overload
    def current_trials_count(
        self, *, split_by_parameters: Literal[True], only_completed: bool = False
    ) -> Int1DArray: ...
    def current_trials_count(
        self, *, split_by_parameters: bool = False, only_completed: bool = False
    ) -> int | Int1DArray:
        """Number of completed trials for this experiment. If split_by_parameters is True, returns a list of counts for each parameter combination."""  # noqa: E501
        studies = self._study_names()
        counts = np.zeros(len(studies), dtype=int)
        db = self.optuna.optuna_db
        if db is None:
            return 0 if not split_by_parameters else counts

        existing_studies = db.list_studies_name()
        for i, study_name in enumerate(studies):
            if study_name in existing_studies:
                study = self.optuna.load_study(study_name)
                counts[i] = study.valid_trials_count(only_completed=only_completed)
        return counts if split_by_parameters else int(sum(counts))

    @overload
    def trials_to_run(self, *, split_by_parameters: Literal[False] = False, ignore_running: bool = False) -> int: ...
    @overload
    def trials_to_run(self, *, split_by_parameters: Literal[True], ignore_running: bool = False) -> Int1DArray: ...
    def trials_to_run(self, *, split_by_parameters: bool = False, ignore_running: bool = False) -> int | Int1DArray:
        """Number of trials left to run for this experiment. If split_by_parameters is True, returns a list of counts for each parameter combination."""  # noqa: E501
        c: Int1DArray = self.current_trials_count(split_by_parameters=True, only_completed=ignore_running)
        c = np.clip(self.n_trials - c, 0, None, dtype=int)
        return c if split_by_parameters else int(c.sum())

    @computed_field
    @cached_property
    def experiment_name(self) -> str:
        """Get the base experiment name without parameter or version information."""
        version = f"v{self.version}" if self.version is not None else ""
        return self.experiment + version

    def _study_names(self) -> list[str]:
        """List of Optuna study names for this experiment, based on the parameter combinations and the number of trials."""  # noqa: E501
        if len(self.parameters_grid) == 0:
            return [self.experiment_name]
        return [self.experiment_name + "-" + param_cfg_name for param_cfg_name in self.parameters_grid.keys()]

    @property
    def file(self) -> Optional[Path]:
        """Path to the YAML file from which this experiment configuration was loaded, if available. This can be used for logging and error reporting purposes."""  # noqa: E501
        return self._file

    @classmethod
    def _read_file(
        cls, file: str | Path, strict: Optional[bool] = None
    ) -> tuple[ExperimentHeader, YamlDocumentWithFile]:
        yaml_docs = YamlDocument.read_file(file)
        if len(yaml_docs) != 2:
            raise InvalidDocumentCountError(expected=2, actual=len(yaml_docs), file=file)
        exp = yaml_docs[0].validate(ExperimentHeader, strict=strict)
        exp._file = yaml_docs[0].file
        return exp, yaml_docs[1]

    @classmethod
    def check_file(cls, file: str | Path, model: type, strict: Optional[bool] = None) -> bool:
        """Check if the given experiment configuration file is valid according to the schema.
        If the file is invalid prints a detailed error message with the validation errors and returns False. Otherwise, returns True.

        Parameters
        ----------
        file : str | Path
            Path to the experiment configuration YAML file.

        model : type
            Pydantic model class to validate the experiment configuration against. This should be the same model that will be used to load the experiment with `load_experiment`.

        strict : Optional[bool], default None
            Whether to use strict validation. If None, uses the default behavior of the model validation (which is strict for BaseModel and non-strict for other types).

        Returns
        -------
        bool
            True if the file is valid, False otherwise.
        """  # noqa: E501
        console = Console(highlight=False)

        try:
            exp, yaml_doc = cls._read_file(file, strict=strict)
        except FileNotFoundError:
            console.print(f"[bold][red]Experiment configuration file not found[/red][/bold]: {file}")
            return False
        except InvalidDocumentCountError as e:
            console.print(f"[bold][red]{e}[/red][/bold]: {file}")
            return False
        except ValidationError as e:
            msg = f"[bold][red]Invalid experiment header[/red][bold]: {file}\n"
            msg += pretty_validation_error_msg(e, ExperimentHeader)
            console.print(msg)
            return False

        for study_name, parameters in exp.parameters_grid.items():
            study = exp.optuna.load_study(study_name, temp_storage=True)
            trial = study.ask(fixed_parameters=parameters)
            with TrialContext(trial):
                try:
                    yaml_doc.validate(model, strict=strict)
                except ValidationError as e:
                    file_link = f"[link=file://{str(yaml_doc.file.absolute())}]{file}[/link]"
                    msg = f"[bold][red]Invalid experiment configuration[/red][/bold]: {file_link} with parameter(s):\n"
                    for k, v in parameters.items():
                        msg += f"\t${k}={repr(v)}\n"
                    msg += "\n" + pretty_validation_error_msg(e, model)
                    console.print(msg)
                    return False
        return True

    @classmethod
    def load_header(cls, file: str | Path) -> ExperimentHeader:
        """Get the number of remaining trials to run for the given experiment configuration file. This is calculated as the total number of trials minus the number of completed trials."""  # noqa: E501
        exp, _ = cls._read_file(file)
        return exp

    @classmethod
    def load_experiment(
        cls,
        file: str | Path,
        model: type,
        strict: Optional[bool] = None,
        header_override: dict | None = None,
        override: dict | None = None,
        param_grid_id: int | None = None,
    ) -> ExperimentRunFactory:
        """Load an experiment configuration from a YAML file and return an ExperimentRunFactory for executing the experiment. Raises an exception if the file is invalid.

        The Yaml file should contains two documents:
            - The first document should be a valid ExperimentCfg, which defines the experiment settings and the hyperparameter grid.
            - The second document should be the experiment configuration, which will be validated against the given model. This document can refer to the parameters defined in the first document using the syntax '$<parameter_name>'.
        """  # noqa: E501
        header, yaml_doc = cls._read_file(file, strict=strict)
        if header_override is not None:
            header = header.model_copy(update=header_override)
        if header.trials_to_run() == 0:
            if not header.test_debug:
                raise NoTrialsToRunError(header, file)
            else:
                console = Console()
                console.print(
                    f"[yellow][bold]Warning:[/bold] All trials for this experiment have already been completed. No remaining trials to run for experiment defined in {file}.[/yellow]"  # noqa: E501
                )
        return ExperimentRunFactory(header, yaml_doc, model, override, param_grid_id=param_grid_id)


class ExperimentRunFactory[T: BaseModel]:
    def __init__(
        self,
        header: ExperimentHeader,
        yaml: YamlDocument,
        model: type[T],
        yaml_override: dict | None = None,
        param_grid_id: int | None = None,
    ):
        self.header = header
        self.model = model
        self.yaml = yaml
        self.yaml_override = yaml_override
        self.param_grid_id = param_grid_id
        self.__ctx_token: Optional[Token[Optional[ExperimentRun]]] = None

    def next_run(self) -> Optional[ExperimentRun[T]]:
        """Get the next experiment run to execute, based on the current trial counts and parameter combinations. Returns None if all trials have been completed."""  # noqa: E501

        cfg = self.header
        param_grid_id = self.param_grid_id
        study_names = cfg._study_names()

        if param_grid_id is None:
            # If param_grid_id is None, find the first parameter combination that has remaining trials to run.
            for i, study_name in enumerate(study_names):
                study = cfg.optuna.load_study(study_name, temp_storage=self.header.test_debug)
                if study.valid_trials_count(only_completed=False) < cfg.n_trials:
                    param_grid_id = i
                    break
            else:
                return None
        else:
            # If param_grid_id is provided, check if it is valid and if there are remaining trials to run
            if param_grid_id < 0 or param_grid_id >= len(study_names):
                raise ValueError(
                    f"Invalid param_grid_id {param_grid_id}. Must be between 0 and {len(study_names) - 1}."
                )
            study_name = study_names[param_grid_id]
            study = cfg.optuna.load_study(study_name, temp_storage=self.header.test_debug)
            if study.valid_trials_count(only_completed=False) >= cfg.n_trials:
                return None

        # Get the next trial with the appropriate fixed parameters
        trial_name, trial_params = list(cfg.parameters_grid.items())[param_grid_id]
        trial = study.ask(fixed_parameters=trial_params)
        with TrialContext(trial):
            # Parse the model and samples run hyperparameters values according to the current trial
            run_cfg = self.yaml.validate(self.model)  # <- HyperParam values are sampled by validators in here
            if self.yaml_override is not None:
                run_cfg = run_cfg.model_copy(update=self.yaml_override)
            return ExperimentRun(self.header, run_cfg, study, trial, trial_name)
        return None

    def __enter__(self) -> ExperimentRun[T]:
        exp_run = self.next_run()
        if exp_run is None:
            raise RuntimeError("All trials for this experiment have already been completed.")
        self.__ctx_token = _current_experiment.set(exp_run)
        exp_run.init()
        return exp_run

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        if self.__ctx_token is not None:
            exp_run = self.__ctx_token.var.get()
            assert exp_run is not None

            if exc_val is None:
                exp_run.finish(state="success")
            elif isinstance(exc_val, optuna.exceptions.TrialPruned):
                exp_run.finish(state="aborted")
            else:
                # Print the exception in red in the console
                console = Console()
                console.print_exception(width=180, extra_lines=3, show_locals=False, theme="monokai")
                exp_run.finish(state="failed")

            _current_experiment.reset(self.__ctx_token)
            self.__ctx_token = None


####################################
#   --- Experiment ---   #
####################################
class ExperimentRun[T: BaseModel]:
    """Context manager for setting the current experiment. This is used internally by the OptunaCfg to manage the experiment context during hyperparameter optimization."""  # noqa: E501

    def __init__(self, exp: ExperimentHeader, cfg: T, study: optuna.study.Study, trial: optuna.Trial, trial_name: str):
        self.header = exp
        self.cfg = cfg
        self._trial_name = trial_name
        self.study = study
        self.trial = trial
        self.logger: Optional[WandbLogger] = None

    @classmethod
    def current(cls) -> Optional[ExperimentRun[T]]:
        """Get the current experiment from the context."""
        return _current_experiment.get()

    @property
    def parameters_grid(self) -> dict[str, Any]:
        """Get the current parameter combination for this run."""
        if len(self.header.parameters_grid) == 0:
            return {}
        return self.header.parameters_grid[self._trial_name]

    @property
    def trial_name(self) -> str:
        """Get the experiment name for this run, based on the experiment configuration and the parameter combination."""  # noqa: E501
        return self.header.experiment_name + ("__" + self._trial_name if self._trial_name else "")

    @property
    def run_id(self) -> int:
        """Get the current run ID from the trial user attributes. This can be used to differentiate between multiple runs of the same trial (e.g. for different random seeds)."""  # noqa: E501
        return self.trial.user_attrs.get("ID", self.trial.number)

    @property
    def run_name(self) -> str:
        return self.trial_name + f"-{self.run_id:02d}"

    def init(self) -> None:
        # Init logger
        config = self.cfg.model_dump()
        config["parameters_grid"] = self.parameters_grid
        config["EXP"] = self.header.experiment_name
        config["TRIAL"] = self.trial_name
        self.logger = WandbLogger(
            name=self.run_name,
            tags=self.header.tags,
            group=self.header.topic,
            project="Test & Debug" if self.header.test_debug else self.header.project,
            config=config,
            log_model=self.header.log_model,
            save_dir="tmp",
        )

        # Log config artifact
        config_artifact = wandb.Artifact(self.run_name.replace(" ", "_"), type="hyper-parameters")
        yaml = YAML()
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as tmp:
            yaml.dump(self.cfg.model_dump(), tmp)
            config_artifact.add_file(tmp.name, name="config.yaml")
        if self.header.file is not None:
            config_artifact.add_file(str(self.header.file.absolute()), name="experiment.yaml")
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as tmp:
            yaml.dump(self.parameters_grid, tmp)
            config_artifact.add_file(tmp.name, name="parameters_grid.yaml")
        self.logger.experiment.log_artifact(config_artifact)

        console = Console()
        console.print(f"[purple]=== Starting experiment run: [bold]{self.run_name}[/bold] ===[/purple]")
        console.print("\t[bold]Parameter Grid:[/bold]")
        for k, v in self.parameters_grid.items():
            console.print(f"\t\t{k}: {v}")
        console.print("\t[bold]Optuna Parameter:[/bold]")
        for k, v in self.trial.params.items():
            console.print(f"\t\t{k}: {v}")
        console.print("\t[bright_black]-- -- -- -- -- -- -- -- -- --[/bright_black]")

    def tell(self, values: float | Sequence[float] | None = None):
        self.study.tell(self.trial, values=values, state=optuna.trial.TrialState.COMPLETE, skip_if_finished=True)

    def finish(self, state: Literal["success", "failed", "aborted"]) -> None:
        """Finish the current trial with the given value and state. This should be called at the end of each trial to report the results to Optuna."""  # noqa: E501
        console = Console()
        match state:
            case "success":
                state_ = optuna.trial.TrialState.COMPLETE
                exit_code = 0
                console.print(f"[green]=== Run {self.run_name} [bold] COMPLETED [/bold] ===[/green]")
            case "failed":
                state_ = optuna.trial.TrialState.FAIL
                exit_code = 10
                console.print(f"[red]=== Run {self.run_name} [bold] FAILED [/bold] ===[/red]")
            case "aborted":
                state_ = optuna.trial.TrialState.PRUNED
                exit_code = 1
                console.print(f"[yellow]=== Run {self.run_name} [bold] PRUNED [/bold] ===[/yellow]")
            case _:
                state_ = None
        self.study.tell(self.trial, state=state_, skip_if_finished=True)

        if self.logger is not None:
            self.logger.finalize(status=state)
            wandb.finish(exit_code=exit_code)
        console.print("\t[bright_black]---------------------------------------------------------------[/bright_black]")

    def pruning_callback(self, monitor: str) -> Callback:
        """Optuna pruning callback to be called at the end of each epoch during training. This will report the intermediate value to Optuna and check if the trial should be pruned."""  # noqa: E501
        from optuna.integration import PyTorchLightningPruningCallback  # type: ignore

        if self.header.optuna.pruner is not None:
            return PyTorchLightningPruningCallback(self.trial, monitor=monitor)
        else:
            return Callback()  # No-op callback


_current_experiment: ContextVar[Optional[ExperimentRun]] = ContextVar("_current_experiment", default=None)


####################################
#   --- Experiment Cfg Base Model ---   #
####################################
class ExpCfgBaseModel(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    @classmethod
    def __init_subclass__(cls, hyperparams: tuple[str, ...] | None = None, **kwargs):
        super().__init_subclass__(**kwargs)

    @classmethod
    def __pydantic_init_subclass__(cls, hyperparams: tuple[str, ...] | None = None, **kwargs):
        from .optuna import attach_hyperparameter_validators

        super().__pydantic_init_subclass__(**kwargs)

        attach_hyperparameter_validators(cls, fields=hyperparams)
        cls.model_rebuild(force=True)

    @model_validator(mode="before")
    @classmethod
    def resolve_experiment_variable(cls, data: Any) -> Any:
        """
        Intercept the validation process to resolve any terminal string fields that start with '$' as references to the current experiment's parameter grid.
        """  # noqa: E501

        for k, v in data.items():
            if isinstance(v, str) and v.startswith("$"):
                var_name = v[1:]  # Strip the '$'
                value = current_trial().user_attrs.get("fixed_params", {}).get(var_name, ...)
                if value is ...:
                    raise ValueError(f"Parameter '{var_name}' was not defined in the parameter grid.")
                data[k] = value

        return data

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler) -> JsonSchemaValue:
        """
        Updates the JSON schema strictly for terminal primitive fields at the top level.
        """
        json_schema = handler(core_schema)  # type: ignore
        json_schema = handler.resolve_ref_schema(json_schema)

        # Inject the schema into the root '$defs'
        VAR_TYPE_NAME, VAR_REF = handler.generate_json_schema.get_cache_defs_ref_schema("ExpParameterRef")
        handler.generate_json_schema.definitions[VAR_TYPE_NAME] = {
            "type": "string",
            "pattern": r"^\$[a-zA-Z_]\w*$",
            "description": "A reference to a parameter defined in the experiment's parameter grid. The variable name should be prefixed with '$'.",  # noqa: E501
        }

        # Modify model schema
        if "properties" in json_schema:
            for _, field_schema in json_schema["properties"].items():
                if not isinstance(field_schema, dict):
                    continue

                if "anyOf" not in field_schema:
                    # Extract original constraints (excluding title)
                    original_type = {
                        k: field_schema.pop(k)
                        for k in list(field_schema.keys())
                        if k not in ("title", "description", "default")
                    }

                    # Convert field into an anyOf union
                    field_schema["anyOf"] = [original_type, VAR_REF]
                elif VAR_REF not in field_schema["anyOf"]:
                    # Append choice string option to top-level optional/union primitives
                    field_schema["anyOf"].append(VAR_REF)
        return json_schema
