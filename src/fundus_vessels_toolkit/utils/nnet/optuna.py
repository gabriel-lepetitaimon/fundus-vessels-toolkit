from __future__ import annotations

import re
import typing
from abc import abstractmethod
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Optional, Self

import optuna
import yaml
from optuna.trial import Trial
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


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

    n_ei_candidates: int = Field(default=8)
    """Number of candidates for the expected improvement in the TPE sampler. Default is 8."""

    def create_sampler(self) -> optuna.samplers.TPESampler:
        return optuna.samplers.TPESampler(n_startup_trials=self.n_startup_trials, n_ei_candidates=self.n_ei_candidates)


class GPSamplerCfg(BaseSamplerCfg):
    type: Literal["GP"] = "GP"

    n_startup_trials: int = Field(default=4)
    """Number of startup trials for the GP sampler. Default is 4."""

    def create_sampler(self) -> optuna.samplers.GPSampler:
        return optuna.samplers.GPSampler(n_startup_trials=self.n_startup_trials)


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


SAMPLER_NAME = Literal["TPE", "Random", "NSGA", "GP"]
type AnySamplerCfg = Annotated[
    TPESamplerCfg | RandomSamplerCfg | NSGASamplerCfg | GPSamplerCfg, Field(discriminator="type")
]


def samplers_by_name(sampler_type):
    if not isinstance(sampler_type, str):
        return sampler_type
    match sampler_type:
        case "TPE":
            return TPESamplerCfg()
        case "Random":
            return RandomSamplerCfg()
        case "NSGA":
            return NSGASamplerCfg()
        case "GP":
            return GPSamplerCfg()
    raise ValueError(f"Unsupported sampler type: {sampler_type}")


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

    n_startup_trials: int = Field(default=5)
    """Pruning is disabled until the given number of trials finish in the same study. Default is 5."""

    n_warmup_epochs: int = Field(default=0)
    """Pruning is disabled until the trial exceeds the given number of epoch."""

    interval_epochs: int = Field(default=1)
    """Interval in number of epochs between the pruning checks, offset by the warmup epochs. If no value has been reported at the time of a pruning check, that particular check will be postponed until a value is reported."""  # noqa: E501

    n_min_trials: int = Field(default=3)
    """Minimum number of reported trial results at an epoch to judge whether to prune. If the number of reported intermediate values from all trials at the current epoch is less than n_min_trials, the trial will not be pruned."""  # noqa: E501

    def create_pruner(self) -> optuna.pruners.MedianPruner:
        return optuna.pruners.MedianPruner(
            n_startup_trials=self.n_startup_trials,
            n_warmup_steps=self.n_warmup_epochs,
            interval_steps=self.interval_epochs,
            n_min_trials=self.n_min_trials,
        )


class SuccessiveHalvingPrunerCfg(BasePrunerCfg):
    type: Literal["SuccessiveHalving"] = "SuccessiveHalving"

    min_epoch: int | Literal["auto"] = Field(default="auto")
    """Min epoch for the successive halving pruner. Default is "auto", which sets min_epoch to the minimum epoch in the study's trials or 1 if there are no completed trials."""  # noqa: E501

    reduction_factor: int = Field(default=4)
    """Reduction factor for the successive halving pruner. Default is 4."""

    min_early_stopping_rate: int = Field(default=0)
    """Minimum early stopping rate for the successive halving pruner. Default is 0."""

    bootstrap_count: int = Field(default=0)
    """Number of bootstrap trials for the successive halving pruner. Default is 0."""

    def create_pruner(self) -> optuna.pruners.SuccessiveHalvingPruner:
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=self.min_epoch,
            reduction_factor=self.reduction_factor,
            min_early_stopping_rate=self.min_early_stopping_rate,
            bootstrap_count=self.bootstrap_count,
        )


PRUNER_NAME = Literal["Median", "SuccessiveHalving", None]
type AnyPrunerCfg = Annotated[MedianPrunerCfg | SuccessiveHalvingPrunerCfg, Field(discriminator="type")]


def pruners_by_name(pruner_type):
    if not isinstance(pruner_type, str):
        return pruner_type
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
    ...     "storage": "sqlite:///optuna.db",
    ...     "sampler": {"type": "TPE"},
    ...     "direction": "minimize"}
    ... )

    """

    model_config = ConfigDict(use_attribute_docstrings=True, frozen=True)

    storage: Optional[str] = Field(default="sqlite:///tmp/optuna.db", pattern=r"^(sqlite|postgresql|mysql)://")
    """Optuna storage URL. If None, use a in-memory non-persistent storage. Default is "sqlite:///optuna.db"."""

    sampler: SamplerCfg = Field(default_factory=TPESamplerCfg)
    """Optuna sampler specification."""

    pruner: Optional[PrunerCfg] = Field(default=None)
    """Pruner to use for Optuna. Default is None (no pruning)."""

    direction: Literal["minimize", "maximize"] = Field(default="maximize")
    """Direction of optimization for Optuna. Default is 'minimize'."""

    @property
    def optuna_db(self) -> OptunaDB | None:
        return OptunaDB(storage=self.storage) if self.storage is not None else None

    def load_study(self, study_name: str, temp_storage: bool = False) -> OptunaStudy:
        return OptunaStudy.load(study_name=study_name, cfg=self, temp_storage=temp_storage)


class OptunaDB:
    """Utility class for managing Optuna studies and trials in a database. This class provides methods for retrieving study information, trial results, and other related data from the Optuna storage."""  # noqa: E501

    def __init__(self, storage: str):
        self.storage = storage

    @classmethod
    def ensure_storage_path(cls, storage: str):
        if storage.startswith("sqlite:///"):
            Path(storage[len("sqlite:///") :]).parent.mkdir(parents=True, exist_ok=True)

    def list_studies_name(self) -> list[str]:
        self.ensure_storage_path(self.storage)
        """List all study names in the Optuna storage."""
        return optuna.get_all_study_names(storage=self.storage)


class OptunaStudy(optuna.study.Study):
    """Utility class for managing a specific Optuna study. This class provides methods for retrieving trial information, best parameters, and other related data for a given Optuna study."""  # noqa: E501

    @classmethod
    def load(cls, study_name: str, cfg: OptunaCfg, temp_storage: bool = False) -> Self:
        if not temp_storage and cfg.storage is not None:
            OptunaDB.ensure_storage_path(cfg.storage)

        study = optuna.create_study(
            study_name=study_name,
            storage=None if temp_storage else cfg.storage,
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
        new_id = 0
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
def attach_hyperparameter_validators(cls, fields: tuple[str, ...] | None = None):
    """Attach Optuna hyperparameter parsing validators to the fields of a Pydantic model class based on the field annotations. This function modifies the field annotations in-place to add BeforeValidators that parse hyperparameter search space strings into actual hyperparameter values using the current Optuna trial context."""  # noqa: E501
    infer_hyperparams = fields is None
    if infer_hyperparams:
        fields = tuple(cls.model_fields.keys())

    if fields:
        # 1. Assign each fields to its type group
        for param in fields:
            if param not in cls.model_fields:
                raise ValueError(f"Hyperparameter '{param}' is not defined as a field in the model.")
            field_info = cls.model_fields[param]
            param_type = extract_type(field_info.annotation)

            if param_type is bool:
                field_info.annotation = Annotated[
                    bool, BeforeValidator(optuna_parse_bool, json_schema_input_type=bool | BoolSearchSpace)
                ]  # type: ignore
            elif param_type is int:
                field_info.annotation = Annotated[
                    int, BeforeValidator(optuna_parse_int, json_schema_input_type=int | IntSearchSpace)
                ]  # type: ignore
            elif param_type is float:
                field_info.annotation = Annotated[
                    float, BeforeValidator(optuna_parse_float, json_schema_input_type=float | FloatSearchSpace)
                ]  # type: ignore
            elif (literals := extract_literals(param_type)) is not None:
                literal_re = literal_pattern(literals)
                pattern = rf"^({literal_re})(\s*{ENUM_SYMBOL}\s*({literal_re}))*$"
                field_info.annotation = Annotated[
                    param_type,
                    BeforeValidator(
                        optuna_parse_literal(literals),
                        json_schema_input_type=param_type | Annotated[str, StringConstraints(pattern=pattern)],
                    ),
                ]  # type: ignore

            elif not infer_hyperparams:
                raise TypeError(
                    f"Unsupported type for hyperparameter '{param}'. "
                    "Only bool, int, float and literal types are supported."
                )


_current_trial: ContextVar[Optional[Trial]] = ContextVar("current_trial", default=None)


class TrialContext:
    """Context manager for setting the current Optuna trial in context. This allows the hyperparameter parsing functions to access the current trial and its parameters when parsing hyperparameter search spaces."""  # noqa: E501

    def __init__(self, trial: Trial):
        self.trial = trial
        self.token: Optional[Token[Trial | None]] = None

    def __enter__(self) -> Trial:
        self.token = _current_trial.set(self.trial)
        return self.trial

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token is not None:
            _current_trial.reset(self.token)


def current_trial() -> Trial:
    from .experiment import ExperimentRun

    current_trial = _current_trial.get()
    if current_trial is not None:
        return current_trial

    exp = ExperimentRun.current()
    if exp is None:
        raise RuntimeError(
            "No active experiment run found in context. current_trial() can only be called within an active experiment run context."  # noqa: E501
        )
    return exp.trial


ENUM_SYMBOL = "~"


def optuna_parse_int(value: int | IntSearchSpace, info: ValidationInfo):
    if not isinstance(value, str):
        return value
    if info.field_name is None:
        raise ValueError("Field name must be provided in ValidationInfo for optuna_parse_int.")

    search_space = re.split(r"[:~]", value)
    if len(search_space) == 2:
        low, high = search_space
        step = 1
    elif len(search_space) == 3:
        low, step, high = search_space
    else:
        raise ValueError(f"Invalid search space format: {value}. Expected format is 'low:high' or 'low:step:high'.")
    return current_trial().suggest_int(info.field_name, int(low), int(high), log="~" in value, step=int(step))


type IntSearchSpace = Annotated[str, StringConstraints(pattern=r"^(\+|-)?\d+(:|~)(\+|-)?\d+(?:(:|~)(\+|-)?\d+)?$")]
# type IntHyperParam = Annotated[int, BeforeValidator(optuna_parse_int, json_schema_input_type=int | IntSearchSpace)]
"""
An integer field accepting either a fixed integer or a string describing an integer search space. 
Search space is defined as "low:high" for uniform sampling or "low~high" for log-uniform sampling.
"""


def optuna_parse_float(value: float | FloatSearchSpace, info: ValidationInfo):
    if not isinstance(value, str):
        return value

    if info.field_name is None:
        raise ValueError("Field name must be provided in ValidationInfo for optuna_parse_float.")

    low, high = re.split(r"[:~]", value)
    return current_trial().suggest_float(info.field_name, float(low), float(high), log="~" in value)


type FloatSearchSpace = Annotated[
    str, StringConstraints(pattern=r"^\d+(\.\d+)?(e[+-]?\d+)?(:|~)\d+(\.\d+)?(e[+-]?\d+)?$")
]
# type FloatHyperParam = Annotated[
#     float, BeforeValidator(optuna_parse_float, json_schema_input_type=float | FloatSearchSpace)
# ]
"""
A float field accepting either a fixed float or a string describing a float search space.
Search space is defined as "low:high" for uniform sampling or "low~high" for log-uniform sampling.
"""


def optuna_parse_bool(value: bool | BoolSearchSpace, info: ValidationInfo):
    if not isinstance(value, str):
        return value
    if info.field_name is None:
        raise ValueError("Field name must be provided in ValidationInfo for optuna_parse_bool.")

    if not re.match(r"^(?:([tT]rue\s*\|\s*[fF]alse)|([fF]alse\s*\|\s*[tT]rue))$", value):
        raise ValueError(f"Invalid bool hyper-parameter format: {value}.")
    return current_trial().suggest_int(info.field_name, 0, 1) == 1


type BoolSearchSpace = Annotated[
    str, StringConstraints(pattern=r"^(?:([tT]rue\s*\|\s*[fF]alse)|([fF]alse\s*\|\s*[tT]rue))$")
]
# type BoolHyperParam =
"""
A boolean field accepting either a fixed boolean or a string describing a boolean search space. 
Search space is defined as "low:high" for uniform sampling or "low~high" for log-uniform sampling.
"""


def optuna_parse_literal(literal_type, to_list: bool = False):
    def parser(value, info: ValidationInfo):
        if not isinstance(value, str):
            return value

        if ENUM_SYMBOL not in value:
            if to_list and isinstance(value, str):
                value = [v.strip() for v in value.split(",")]
            return value
        assert info.field_name is not None, "Field has no name."

        values = value.split(ENUM_SYMBOL)
        adapter = TypeAdapter(list[literal_type] if to_list else literal_type)
        if to_list:
            values = [yaml.safe_load(v.strip()) for v in values]
        values_ = []
        for v in values:
            try:
                values_.append(adapter.validate_python(v))
            except ValidationError as e:
                raise ValueError(
                    f"Invalid value in search space: {v}. Valid values are list of: {extract_literals(literal_type)}"
                ) from None
        return values_[current_trial().suggest_int(info.field_name, 0, len(values_) - 1)]

    return parser


def literal_pattern(literals) -> str:
    if None in literals:
        literals = [v for v in literals if v is not None] + ["null"]
    return "|".join(re.escape(str(v)) for v in literals)


def extract_literals(literal_type) -> list[Any] | None:
    literal_type = extract_type(literal_type)
    if (origin := typing.get_origin(literal_type)) is not None:
        if origin is typing.Literal:
            return list(typing.get_args(literal_type))
        if origin is typing.Union:
            union_literals = []
            for t in typing.get_args(literal_type):
                literals = extract_literals(t)
                if literals is None:
                    return None
                union_literals.extend(literals)
            return union_literals
    return None


def extract_type(type_):
    """Recursively extract the base type from a potentially nested Annotated or alias type."""
    if hasattr(type_, "__value__"):
        return extract_type(type_.__value__)
    if (origin := typing.get_origin(type_)) is not None:
        if origin is typing.Annotated:
            return extract_type(typing.get_args(type_)[0])
        if origin is typing.Union and len(union := typing.get_args(type_)) == 1:
            return extract_type(union[0])
    return type_


# def LiteralHyperParam(literal_type):
#     """Annotation for a hyperparameter that can be either a fixed literal value or a string describing a categorical search space.
#     Search space is defined as "value1~value2~value3" for categorical sampling.

#     Parameters
#     ----------
#     literal_type :
#         Literal type defining the allowed fixed values for the hyperparameter.

#     Examples
#     --------
#     >>> CustomLiteral = Literal["a", "b", None]
#     >>> test_version: Annotated[CustomLiteral, LiteralHyperParam(CustomLiteral)] = Field(default=None)

#     """  # noqa: E501
#     return BeforeValidator(
#         optuna_parse_literal(literal_type),
#         json_schema_input_type=literal_type | _LiteralSearchSpace[literal_type],
#     )


class _ListLiteralSearchSpace:
    @classmethod
    def pattern(cls, literal_type) -> str:
        literals = extract_literals(literal_type)
        if literals is None:
            raise TypeError("Literal type expected for _ListLiteralSearchSpace.")
        literal_re = literal_pattern(literals)
        array_re = rf"\s*\[\s*({literal_re})\s*(?:,\s*({literal_re})\s*)*\]"
        return rf"({array_re})(\s*{ENUM_SYMBOL}\s*({array_re}))*"

    @classmethod
    def __class_getitem__(cls, T):
        return Annotated[str, StringConstraints(pattern=rf"^{cls.pattern(T)}$")]


def ListLiteralHyperParam(literal_type):
    """Annotation for a hyperparameter that can be either a fixed list of literal values or a string describing a categorical search space over lists of literals.

    Search space is defined as "[value1, value2]~[value3, value4]~[value5]" for categorical sampling over lists of literals.

    Parameters
    ----------
    literal_type :
        Literal type defining the allowed fixed values for the elements of the list.

    Examples
    --------
    >>> CustomLiteral = Literal["a", "b", None]
    >>> training_set: Annotated[list[CustomLiteral], ListLiteralHyperParam(CustomLiteral)] = Field(default=[None])
    """  # noqa: E501
    return BeforeValidator(
        optuna_parse_literal(literal_type, to_list=True),
        json_schema_input_type=literal_type | list[literal_type] | _ListLiteralSearchSpace[literal_type],
    )


_BoolHyperParam = Annotated[bool, BeforeValidator(optuna_parse_bool, json_schema_input_type=bool | BoolSearchSpace)]


def BoolDefaultValidator[T](return_type: type[T], default_factory: Optional[Callable[[], T]] = None) -> BeforeValidator:
    def validator(value: T | bool, info: ValidationInfo) -> Optional[T]:
        if isinstance(value, str):
            value = optuna_parse_bool(value, info)
        if value is True:
            if default_factory is None:
                return return_type() if isinstance(return_type, type) else return_type
            return default_factory()
        elif value is False:
            return None
        return value

    return BeforeValidator(validator, json_schema_input_type=_BoolHyperParam | Optional[return_type])
