"""Portable research records linking scores, their meaning and their evaluation rows."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite


def required_text(**fields: str) -> None:
    for name, value in fields.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} needs a definition or reference")


@dataclass(frozen=True)
class Continuation:
    """A common complete-play utility, fixed perspective and future policies after a forced move."""
    utility: str
    environment: str
    first_seat: str
    second_seat: str
    horizon: str
    perspective: str
    discount: float
    rollouts: int | None  # None means exact expectation.

    def __post_init__(self) -> None:
        required_text(utility=self.utility, environment=self.environment, first=self.first_seat,
                      second=self.second_seat, horizon=self.horizon, perspective=self.perspective)
        if not 0 <= self.discount <= 1 or (self.rollouts is not None and
                (type(self.rollouts) is not int or self.rollouts < 1)):
            raise ValueError("invalid discount or simulations per candidate")


@dataclass(frozen=True)
class RecoveryContract:
    domain: str
    source: str
    semantics: str
    collection_access: str
    deployment_access: str
    population: str
    candidates: str
    decoder: str
    ties: str
    normalization: str
    units: str
    split: str
    scope: str
    game: bool = False
    opponent_panel: str | None = None
    temperature: float | None = None
    base_weights: str | None = None
    continuation: Continuation | None = None

    def __post_init__(self) -> None:
        required_text(domain=self.domain, source=self.source, collection_access=self.collection_access,
                      deployment_access=self.deployment_access, population=self.population,
                      candidates=self.candidates, ties=self.ties, normalization=self.normalization,
                      units=self.units, split=self.split, scope=self.scope)
        if self.semantics not in {"behavior", "preference", "return", "internal"}:
            raise ValueError("unknown score meaning")
        if self.decoder not in {"direct maximum", "minimum energy", "temperature sampling",
                                "VNM maximum", "VNM temperature sampling"}:
            raise ValueError("unknown decoder")
        if type(self.game) is not bool:
            raise ValueError("game must be a boolean")
        if self.game:
            required_text(opponent_panel=self.opponent_panel)
        if self.decoder.startswith("VNM") or self.semantics == "return":
            if not isinstance(self.continuation, Continuation):
                raise ValueError("expected utility and returns need a complete continuation")
        if "sampling" in self.decoder:
            if self.temperature is None or not isfinite(self.temperature) or self.temperature <= 0:
                raise ValueError("sampling needs a finite positive temperature")
            required_text(base_weights=self.base_weights)
        elif self.temperature is not None or self.base_weights is not None:
            raise ValueError("sampling settings need a sampling decoder")


@dataclass(frozen=True)
class ValueArtifact:
    """A versioned JSON score table, fitted weights or reference, with provenance and uncertainty."""
    contract: RecoveryContract
    method: str
    provenance: str
    uncertainty: str
    payload: dict

    def __post_init__(self) -> None:
        required_text(method=self.method, provenance=self.provenance, uncertainty=self.uncertainty)
        if not isinstance(self.contract, RecoveryContract) or not isinstance(self.payload, dict) or not self.payload:
            raise ValueError("a contract and score data or a reference are required")
        # Round-trip equality excludes lossy inputs such as integer object keys or tuples.
        if json.loads(json.dumps(self.payload, allow_nan=False)) != self.payload:
            raise ValueError("payload needs lossless JSON types")

    def to_json(self) -> str:
        return json.dumps({"version": 1, **asdict(self)}, allow_nan=False, ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, data: str) -> ValueArtifact:
        obj = json.loads(data)
        if obj.pop("version") != 1:
            raise ValueError("unsupported artifact version")
        contract = obj.pop("contract")
        if contract.get("continuation") is not None:
            contract["continuation"] = Continuation(**contract["continuation"])
        return cls(contract=RecoveryContract(**contract), **obj)

    @property
    def artifact_id(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def evaluation_key(self) -> dict:
        """Include the content identifier and complete contract on independently readable rows."""
        return {"artifact_id": self.artifact_id, "method": self.method, **asdict(self.contract)}
