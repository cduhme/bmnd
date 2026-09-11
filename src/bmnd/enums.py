from enum import Enum, StrEnum
from typing import TypeVar


class NoiseModel(StrEnum):
    GAUSSIAN = "gaussian"
    POISSON = "poisson"


class ReferenceScheduleMode(StrEnum):
    OFFICIAL = "official"
    GENERATED = "generated"
    BALANCED = "balanced"
    SPARSE = "sparse"
    OFF = "off"


class CacheMode(StrEnum):
    AUTO = "auto"


class BlockMatchDistance(StrEnum):
    AUTO = "auto"
    SSD = "ssd"
    POISSON_DEVIANCE = "poisson_deviance"
    PEARSON = "pearson"
    ANSCOMBE_SSD = "anscombe_ssd"


class AggregationWeightModel(StrEnum):
    CLASSIC = "classic"
    VARIANCE = "variance"
    RISK = "risk"


class AggregationWeightDomain(StrEnum):
    COEFFICIENT = "coefficient"
    WINDOWED_SYNTHESIS = "windowed_synthesis"


class AggregationWeightScope(StrEnum):
    GROUP = "group"
    PATCH = "patch"


class PoissonVarianceSource(StrEnum):
    OBSERVATION = "observation"
    PILOT = "pilot"


class PoissonHTMatchPolicy(StrEnum):
    FIXED = "fixed"
    REFERENCE_FINITE_COUNT = "reference_finite_count"
    CANDIDATE_STANDARDIZED = "candidate_standardized"


class PoissonWienerGainMode(StrEnum):
    CLASSIC = "classic"
    NOISE_FLOOR = "noise-floor"
    VARIANCE_SCALED = "variance-scaled"


class PoissonGroupMassConservation(StrEnum):
    NONE = "none"
    HT = "ht"
    WIENER = "wiener"
    BOTH = "both"


EnumT = TypeVar("EnumT", bound=Enum)


def coerce_enum(value: EnumT | str, enum_type: type[EnumT], name: str) -> EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        expected = ", ".join(repr(member.value) for member in enum_type)
        raise ValueError(f"Unknown {name}: {value!r} (expected {expected}).") from exc
