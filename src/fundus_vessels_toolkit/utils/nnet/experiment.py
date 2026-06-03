from __future__ import annotations

import itertools
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, overload

import numpy as np
import optuna
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PrivateAttr, ValidationError
from pytorch_lightning.callbacks import Callback
from rich.console import Console

from fundus_toolkits.utils.typing import Int1DArray

from .optuna import OptunaCfg
from .pydantic_yaml import model_validate_yaml_file, pretty_validation_error_msg


def _validate_parameters_grid(value: dict[str, list] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        k, v = list(value.keys()), list(value.values())
        return [dict(zip(k, c, strict=True)) for c in itertools.product(*v)]
    elif isinstance(value, list):
        if len(value) == 0:
            return []
        if not all(isinstance(item, dict) for item in value):
            raise ValueError("All items in the parameters_grid list must be dictionaries.")
        keys0 = set(value[0].keys())
        for item in value:
            if set(item.keys()) != keys0:
                raise ValueError("All dictionaries in the parameters_grid list must have the same keys.")
        return value
    else:
        raise ValueError("parameters_grid must be either a dict of lists or a list of dicts.")


class ExperimentCfg(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True, frozen=True)

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

    parameters_grid: Annotated[
        list[dict[str, Any]],
        BeforeValidator(_validate_parameters_grid, json_schema_input_type=dict[str, list] | list[dict[str, Any]]),
    ] = Field(default_factory=list)
    """Grid of parameters values to explore. This experiment will be run for each combination of parameters.
    The value of the parameters defined here can be referred to in the configuration file as '%<parameter_name>'"""  # noqa: E501

    _parameters_hashes: list[str] = PrivateAttr(default_factory=list)

    optuna: OptunaCfg
    """Optuna configuration for hyperparameter optimization."""

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
        counts = np.zeros(len(self.parameters_grid), dtype=int)
        db = self.optuna.optuna_db
        if db is None:
            return 0 if not split_by_parameters else counts

        existing_studies = db.list_studies_name()
        for i, study_name in enumerate(self._study_names()):
            if study_name in existing_studies:
                study = self.optuna.load_study(study_name)
                counts[i] = study.valid_trials_count(only_completed=only_completed)
        return counts if split_by_parameters else sum(counts)

    def trials_to_run(self, *, split_by_parameters: bool = False, ignore_running: bool = False) -> int | list[int]:
        """Number of trials left to run for this experiment. If split_by_parameters is True, returns a list of counts for each parameter combination."""  # noqa: E501
        c = self.current_trials_count(split_by_parameters=False, only_completed=ignore_running)
        c = np.clip(self.n_trials - c, 0, None)
        return c if split_by_parameters else c.sum()

    def next_run(self) -> Optional[ExperimentRun]:
        """Get the next experiment run to execute, based on the current trial counts and parameter combinations. Returns None if all trials have been completed."""  # noqa: E501
        for i, study_name in enumerate(self._study_names()):
            study = self.optuna.load_study(study_name)
            if study.valid_trials_count(only_completed=False) < self.n_trials:
                return ExperimentRun(self, i, study, study.ask(fixed_parameters=self.parameters_grid[i]))
        return None

    @property
    def parameters_hashes(self) -> list[str]:
        """List of hashes for each parameter combination. This can be used to uniquely identify trials with specific parameter combinations."""  # noqa: E501
        if not self._parameters_hashes:

            def parameter_hash(params: dict[str, Any]) -> str:
                return "|".join(str(params[k]) for k in sorted(params.keys()))

            self._parameters_hashes = [parameter_hash(params) for params in self.parameters_grid]
        return self._parameters_hashes

    def _study_names(self) -> list[str]:
        """List of Optuna study names for this experiment, based on the parameter combinations and the number of trials."""  # noqa: E501
        return [self.experiment + "-" + param_hash for param_hash in self.parameters_hashes]

    @classmethod
    def check_file(cls, file: str | Path, model: type, strict: Optional[bool] = None) -> bool:
        """Check if the given experiment configuration file is valid according to the schema. Raises an exception if the file is invalid."""  # noqa: E501
        console = Console(highlight=False)
        file = Path(file)
        if not file.exists():
            console.print(f"[bold][red]Experiment configuration file not found[/red][/bold]: {file}")
            return False
        try:
            exp = model_validate_yaml_file(file, ExperimentCfg, strict=strict)
        except ValidationError as e:
            msg = f"[bold][red]Invalid experiment header[/red][bold]: {file}\n"
            msg += pretty_validation_error_msg(e, ExperimentCfg)
            console.print(msg)
            return False

        for i, study_name in enumerate(exp._study_names()):
            study = exp.optuna.load_study(study_name, ram_storage=True)
            run = ExperimentRun(exp, i, study, study.ask(fixed_parameters=exp.parameters_grid[i]))
            with run:
                try:
                    model_validate_yaml_file(file, model, document_id=1, strict=strict)
                except ValidationError as e:
                    file_link = f"[link=file://{str(file.absolute())}]{file}[/link]"
                    msg = f"[bold][red]Invalid experiment configuration[/red][/bold]: {file_link} with parameter(s):\n"
                    for k, v in exp.parameters_grid[i].items():
                        msg += f"\t${k}={repr(v)}\n"
                    msg += "\n" + pretty_validation_error_msg(e, model)
                    console.print(msg)
                    return False
        return True


####################################
#   --- Experiment ---   #
####################################
class ExperimentRun:
    """Context manager for setting the current experiment. This is used internally by the OptunaCfg to manage the experiment context during hyperparameter optimization."""  # noqa: E501

    def __init__(
        self,
        cfg: ExperimentCfg,
        param_config_id: int,
        study: optuna.study.Study,
        trial: optuna.Trial,
    ):
        self.cfg = cfg
        self.param_config_id = param_config_id
        self.study = study
        self.trial = trial
        self.__ctx_token: Optional[Token[Optional[ExperimentRun]]] = None

    def __enter__(self) -> None:
        self.__ctx_token = _current_experiment.set(self)

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        if self.__ctx_token is not None:
            _current_experiment.reset(self.__ctx_token)
            self.__ctx_token = None

    @classmethod
    def current(cls) -> Optional[ExperimentRun]:
        """Get the current experiment from the context."""
        return _current_experiment.get()

    def pruning_callback(self, monitor: str) -> Callback:
        """Optuna pruning callback to be called at the end of each epoch during training. This will report the intermediate value to Optuna and check if the trial should be pruned."""  # noqa: E501
        from optuna.integration import PyTorchLightningPruningCallback

        if self.cfg.optuna.pruner is not None:
            return PyTorchLightningPruningCallback(self.trial, monitor=monitor)
        else:
            return Callback()  # No-op callback


_current_experiment: ContextVar[Optional[ExperimentRun]] = ContextVar("_current_experiment", default=None)
