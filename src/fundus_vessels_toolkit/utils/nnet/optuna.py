from __future__ import annotations

import re
from abc import abstractmethod
from contextvars import ContextVar
from typing import Annotated, Any, Callable, Literal, Optional, Self, Sequence, get_args

import optuna
from optuna.distributions import CategoricalChoiceType
from optuna.trial import Trial
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PrivateAttr, StringConstraints, ValidationInfo


##################################
#   --- Optuna Samplers ---      #
##################################
class BaseSamplerCfg(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    type: SAMPLER_NAME
    """Type of the optuna sampler. Must be one of: "Random", "NSGA", "TPE"."""

    @abstractmethod
    def create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create an Optuna sampler based on this configuration."""
        raise NotImplementedError


class TPESamplerCfg(BaseSamplerCfg):
    type: Literal["TPE"] = "TPE"

    n_startup_trials: int = Field(default=4)
    """Number of startup trials for the TPE sampler. Default is 4."""

    def create_sampler(self) -> optuna.samplers.TPESampler:
        return optuna.samplers.TPESampler(warn_independent_sampling=True)


class RandomSamplerCfg(BaseSamplerCfg):
    type: Literal["Random"] = "Random"

    seed: Optional[int] = Field(default=None)
    """Random seed for the random sampler. Default is None (no seed)."""

    def create_sampler(self) -> optuna.samplers.RandomSampler:
        return optuna.samplers.RandomSampler(seed=self.seed)


class NSGASamplerCfg(BaseSamplerCfg):
    type: Literal["NSGA"] = "NSGA"

    def create_sampler(self) -> optuna.samplers.NSGAIIISampler:
        return optuna.samplers.NSGAIIISampler()


SAMPLER_NAME = Literal["TPE", "Random", "NSGA"]
type AnySamplerCfg = Annotated[TPESamplerCfg | RandomSamplerCfg | NSGASamplerCfg, Field(discriminator="type")]


def samplers_by_name(sampler_type: SAMPLER_NAME | AnySamplerCfg) -> AnySamplerCfg:
    match sampler_type:
        case "TPE":
            return TPESamplerCfg()
        case "Random":
            return RandomSamplerCfg()
        case "NSGA":
            return NSGASamplerCfg()
    return sampler_type


type SamplerCfg = Annotated[
    AnySamplerCfg, BeforeValidator(samplers_by_name, json_schema_input_type=SAMPLER_NAME | AnySamplerCfg)
]


##################################
#   --- Optuna Pruners ---      #
##################################
class BasePrunerCfg(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    type: PRUNER_NAME
    """Type of the optuna pruner. Must be one of: "Median", "SuccessiveHalving", None."""

    @abstractmethod
    def create_pruner(self) -> optuna.pruners.BasePruner:
        """Create an Optuna pruner based on this configuration."""
        raise NotImplementedError


class MedianPrunerCfg(BasePrunerCfg):
    type: Literal["Median"] = "Median"

    def create_pruner(self) -> optuna.pruners.MedianPruner:
        return optuna.pruners.MedianPruner()


class SuccessiveHalvingPrunerCfg(BasePrunerCfg):
    type: Literal["SuccessiveHalving"] = "SuccessiveHalving"

    def create_pruner(self) -> optuna.pruners.SuccessiveHalvingPruner:
        return optuna.pruners.SuccessiveHalvingPruner()


PRUNER_NAME = Literal["Median", "SuccessiveHalving", None]
type AnyPrunerCfg = Annotated[MedianPrunerCfg | SuccessiveHalvingPrunerCfg, Field(discriminator="type")]


def pruners_by_name(pruner_type: PRUNER_NAME | AnyPrunerCfg) -> AnyPrunerCfg:
    match pruner_type:
        case "Median":
            return MedianPrunerCfg()
        case "SuccessiveHalving":
            return SuccessiveHalvingPrunerCfg()
    raise ValueError(f"Unsupported pruner type: {pruner_type}")


type PrunerCfg = Annotated[
    AnyPrunerCfg, BeforeValidator(pruners_by_name, json_schema_input_type=PRUNER_NAME | AnyPrunerCfg)
]


################################
#   --- Optuna Config ---      #
################################
class OptunaCfg(BaseModel):
    """Optuna configuration for hyperparameter optimization.

    Examples
    --------
    >>> optuna = Optuna.model_validate({
    ...     "study_name": "my_study",
    ...     "storage": "sqlite:///optuna.db",
    ...     "sampler": {"type": "TPE"},
    ...     "direction": "minimize"}
    ... )

    """

    model_config = ConfigDict(use_attribute_docstrings=True, frozen=True)

    storage: Optional[str] = Field(default=None, pattern=r"^(sqlite|postgresql|mysql)://")
    """Optuna storage URL. If None (by default), use a in-memory non-persistent storage."""

    sampler: SamplerCfg = Field(default_factory=RandomSamplerCfg)
    """Optuna sampler specification."""

    pruner: Optional[PrunerCfg] = Field(default=None)
    """Pruner to use for Optuna. Default is None (no pruning)."""

    direction: Literal["minimize", "maximize"] = Field(default="minimize")
    """Direction of optimization for Optuna. Default is 'minimize'."""

    @property
    def optuna_db(self) -> OptunaDB | None:
        return OptunaDB(storage=self.storage) if self.storage is not None else None

    def load_study(self, study_name: str) -> OptunaStudy:
        return OptunaStudy.load(study_name=study_name, cfg=self)


class OptunaDB:
    """Utility class for managing Optuna studies and trials in a database. This class provides methods for retrieving study information, trial results, and other related data from the Optuna storage."""  # noqa: E501

    def __init__(self, storage: str):
        self.storage = storage

    def list_studies_name(self) -> list[str]:
        """List all study names in the Optuna storage."""
        return optuna.get_all_study_names(storage=self.storage)


class OptunaStudy(optuna.study.Study):
    """Utility class for managing a specific Optuna study. This class provides methods for retrieving trial information, best parameters, and other related data for a given Optuna study."""  # noqa: E501

    @classmethod
    def load(cls, study_name: str, cfg: OptunaCfg) -> Self:
        study = optuna.create_study(
            study_name=study_name,
            storage=cfg.storage,
            sampler=cfg.sampler.create_sampler(),
            pruner=cfg.pruner.create_pruner() if cfg.pruner is not None else None,
            direction=cfg.direction,
            load_if_exists=True,
        )
        study.__class__ = cls
        return study  # type: ignore

    def valid_trials_count(self, only_completed=False):
        from optuna.trial import TrialState as State

        n_valid_trials = 0

        for trial in self.trials:
            if (only_completed and trial.state in (State.COMPLETE, State.PRUNED)) or trial.state != State.FAIL:
                n_valid_trials += 1
        return n_valid_trials

    def new_trial_id(self) -> int:
        """Generate a new unique trial ID for the next trial to be added to the study. This method checks the existing trials in the study and returns an ID that is not currently used by any non-failed trial."""  # noqa: E501
        from optuna.trial import TrialState as State

        used_id = set(trial.user_attrs.get("ID", 0) for trial in self.trials if trial.state != State.FAIL)
        new_id = 1
        while new_id in used_id:
            new_id += 1
        return new_id

    def ask(self, fixed_parameters: Optional[dict[str, Any]] = None) -> Trial:
        trial = super().ask()
        trial.set_user_attr("ID", self.new_trial_id())
        if fixed_parameters is not None:
            trial.set_user_attr("fixed_params", fixed_parameters)
        return trial


##############################################################################################################
# === OPTUNA / PYDANTIC HYPERPARAMETERS ===
##############################################################################################################
def current_trial() -> Trial:
    from .experiment import ExperimentRun

    exp = ExperimentRun.current()
    if exp is None:
        raise RuntimeError(
            "No active experiment run found in context. current_trial() can only be called within an active experiment run context."  # noqa: E501
        )
    return exp.trial


VAR_SYMBOL = r"\$"
VAR_PATTERN = rf"({VAR_SYMBOL}[a-zA-Z_]\w*)"


def optuna_parse_int(value: int | IntSearchSpace, info: ValidationInfo) -> int:
    if isinstance(value, int):
        return value

    if value.startswith(VAR_SYMBOL):
        params = current_trial().user_attrs.get("fixed_params", {}).get(value[1:], ...)
        if params is ...:
            raise ValueError(f"Parameter '{value[1:]}' not found in fixed parameters of the current trial.")
        try:
            return int(params)
        except Exception as e:
            raise ValueError(f"Failed to parse parameter '{value[1:]}' with value '{params}' as int.") from None

    if info.field_name is None:
        raise ValueError("Field name must be provided in ValidationInfo for optuna_parse_int.")

    low, high = re.split(r"[:~]", value)
    return current_trial().suggest_int(info.field_name, int(low), int(high), log="~" in value)


type IntSearchSpace = Annotated[str, StringConstraints(pattern=rf"^(?:(\d+(:|~)\d+)|{VAR_PATTERN})$")]
type IntHyperParam = Annotated[int, BeforeValidator(optuna_parse_int, json_schema_input_type=int | IntSearchSpace)]
"""
An integer field accepting either a fixed integer or a string describing an integer search space. 
Search space is defined as "low:high" for uniform sampling or "low~high" for log-uniform sampling.
"""


def optuna_parse_float(value: float | FloatSearchSpace, info: ValidationInfo) -> float:
    if isinstance(value, (float, int)):
        return float(value)

    if value.startswith(VAR_SYMBOL):
        params = current_trial().user_attrs.get("fixed_params", {}).get(value[1:], ...)
        if params is ...:
            raise ValueError(f"Parameter '{value[1:]}' not found in fixed parameters of the current trial.")
        try:
            return float(params)
        except Exception as e:
            raise ValueError(f"Failed to parse parameter '{value[1:]}' with value '{params}' as float.") from None

    if info.field_name is None:
        raise ValueError("Field name must be provided in ValidationInfo for optuna_parse_float.")

    low, high = re.split(r"[:~]", value)
    return current_trial().suggest_float(info.field_name, float(low), float(high), log="~" in value)


type FloatSearchSpace = Annotated[
    str, StringConstraints(pattern=rf"^(?:(\d+(\.\d+)?(e[+-]?\d+)?(:|~)\d+(\.\d+)?(e[+-]?\d+)?)|{VAR_PATTERN})$")
]
type FloatHyperParam = Annotated[
    float, BeforeValidator(optuna_parse_float, json_schema_input_type=float | FloatSearchSpace)
]
"""
A float field accepting either a fixed float or a string describing a float search space.
Search space is defined as "low:high" for uniform sampling or "low~high" for log-uniform sampling.
"""


def optuna_parse_literal[T: CategoricalChoiceType](valid_values: Sequence[T]) -> Callable[[str | T, ValidationInfo], T]:
    def str_to_literal(value: str) -> T:
        for v in valid_values:
            if re.escape(str(v)) == value:
                return v
        raise ValueError(f"Value '{value}' is not in the list of valid values: {valid_values}.")

    def parser(value: str | T, info: ValidationInfo) -> T:
        value = str(value)
        if value.startswith(VAR_SYMBOL):
            params = current_trial().user_attrs.get("fixed_params", {}).get(value[1:], ...)
            if params is ...:
                raise ValueError(f"Parameter '{value[1:]}' not found in fixed parameters of the current trial.")
            try:
                return str_to_literal(params)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse parameter '{value[1:]}' with value '{params}' as a valid literal ({T})."
                ) from None

        if "|" not in value:
            return str_to_literal(value)
        values = value.split("|")
        assert info.field_name is not None, "Field has no name."
        values_: list[T] = [str_to_literal(v) for v in values]
        return current_trial().suggest_categorical(info.field_name, values_)  # type: ignore

    return parser


def _pattern_from_literal(literal_type) -> str:
    valid_values_re = "|".join(re.escape(str(v)) for v in get_args(literal_type))
    return r"(" + valid_values_re + r")(\|(" + valid_values_re + r"))*"


type LiteralSearchSpace[T] = Annotated[
    T, StringConstraints(pattern=rf"^(?:({_pattern_from_literal(T)})|{VAR_PATTERN})$")
]
type LiteralHyperParam[T] = Annotated[
    T, BeforeValidator(optuna_parse_literal(get_args(T)), json_schema_input_type=str | T)
]
"""
A field bounded to a set of literal values, accepting either a fixed value or a string describing a categorical search space.
Search space is defined as "value1|value2|value3" for categorical sampling.
"""  # noqa: E501
