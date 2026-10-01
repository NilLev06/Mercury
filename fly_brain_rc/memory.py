"""Bounded visual familiarity and transient neural activity.

Familiarity is the only state in this module that can be persisted, and only
an explicit ``save`` or ``load`` touches the filesystem. Motor weights and
short-term target/threat activity are never stored here.
"""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


_FORMAT_VERSION = 2
_DESCRIPTOR_SIZE = 64
_MAX_CAPACITY = 4096
_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_STRENGTH = 20.0


@dataclass(frozen=True)
class FamiliarityEntry:
    descriptor: np.ndarray
    strength: float
    last_seen: float


class FamiliarityMemory:
    """Match visual descriptors and keep the most useful recent views."""

    def __init__(
        self,
        capacity: int = 128,
        match_threshold: float = 0.85,
        decay_rate: float = 0.1,
        strength_increment: float = 1.0,
    ):
        if type(capacity) is not int or not 1 <= capacity <= _MAX_CAPACITY:
            raise ValueError("capacity must be an integer from 1 to 4096")
        if not _finite_number(match_threshold) or not 0.0 < match_threshold <= 1.0:
            raise ValueError("match_threshold must be finite and in (0, 1]")
        if not _finite_number(decay_rate) or decay_rate < 0.0:
            raise ValueError("decay_rate must be finite and nonnegative")
        if not _finite_number(strength_increment) or strength_increment <= 0.0:
            raise ValueError("strength_increment must be finite and positive")
        self.capacity = capacity
        self.match_threshold = float(match_threshold)
        self.decay_rate = float(decay_rate)
        self.strength_increment = float(strength_increment)
        self._entries: list[FamiliarityEntry] = []
        self._saved_ages: list[float] | None = None

    @property
    def entries(self) -> tuple[FamiliarityEntry, ...]:
        """Read-only snapshots of the stored views."""
        return tuple(
            FamiliarityEntry(entry.descriptor.copy(), entry.strength, entry.last_seen)
            for entry in self._entries
        )

    def observe(self, descriptor: np.ndarray, now: float) -> float:
        """Return familiarity in [0, 1] after observing this view."""
        if not _finite_number(now) or now < 0.0:
            raise ValueError("now must be finite and nonnegative")
        vector = np.asarray(descriptor, dtype=np.float64)
        if vector.shape != (_DESCRIPTOR_SIZE,) or not np.isfinite(vector).all():
            raise ValueError("descriptor must contain 64 finite values")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("descriptor must have finite nonzero magnitude")
        vector = vector / norm
        if self._saved_ages is not None:
            self._entries = [
                FamiliarityEntry(entry.descriptor, entry.strength, float(now) - age)
                for entry, age in zip(self._entries, self._saved_ages)
            ]
            self._saved_ages = None

        if self._entries:
            similarities = [float(np.dot(vector, entry.descriptor)) for entry in self._entries]
            best = int(np.argmax(similarities))
            if similarities[best] >= self.match_threshold:
                entry = self._entries[best]
                strength = min(
                    _MAX_STRENGTH,
                    self._decayed_strength(entry, float(now)) + self.strength_increment,
                )
                self._entries[best] = FamiliarityEntry(vector, strength, float(now))
                return 1.0 - math.exp(-strength)

        if len(self._entries) == self.capacity:
            least_useful = min(
                range(len(self._entries)),
                key=lambda index: (
                    self._decayed_strength(self._entries[index], float(now)),
                    self._entries[index].last_seen,
                ),
            )
            del self._entries[least_useful]
        self._entries.append(FamiliarityEntry(vector, min(_MAX_STRENGTH, self.strength_increment), float(now)))
        return 0.0

    def _decayed_strength(self, entry: FamiliarityEntry, now: float) -> float:
        # A loaded store is rebased to the current episode clock before this
        # method is called. Clock restarts cannot create negative elapsed time.
        elapsed = max(0.0, now - entry.last_seen)
        return entry.strength * math.exp(-self.decay_rate * elapsed)

    def save(self, path: Path) -> None:
        """Atomically replace a versioned JSON memory file beside its temp file."""
        path = Path(path)
        if self._saved_ages is None:
            reference = max((entry.last_seen for entry in self._entries), default=0.0)
            ages = [reference - entry.last_seen for entry in self._entries]
        else:
            ages = self._saved_ages
        payload = {
            "version": _FORMAT_VERSION,
            "capacity": self.capacity,
            "match_threshold": self.match_threshold,
            "decay_rate": self.decay_rate,
            "strength_increment": self.strength_increment,
            "entries": [
                {
                    "descriptor": entry.descriptor.tolist(),
                    "strength": entry.strength,
                    "age": age,
                }
                for entry, age in zip(self._entries, ages)
            ],
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, allow_nan=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> "FamiliarityMemory":
        """Load a validated store only when explicitly requested."""
        path = Path(path)
        if path.stat().st_size > _MAX_FILE_BYTES:
            raise ValueError("memory file exceeds size limit")
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("invalid memory JSON") from exc
        if (not isinstance(payload, dict) or type(payload.get("version")) is not int
                or payload["version"] not in (1, _FORMAT_VERSION)):
            raise ValueError("unsupported memory format version")
        if set(payload) != {
            "version", "capacity", "match_threshold", "decay_rate", "strength_increment", "entries"
        }:
            raise ValueError("invalid memory fields")
        try:
            memory = cls(
                capacity=payload["capacity"],
                match_threshold=payload["match_threshold"],
                decay_rate=payload["decay_rate"],
                strength_increment=payload["strength_increment"],
            )
            entries = payload["entries"]
            if not isinstance(entries, list) or len(entries) > memory.capacity:
                raise ValueError("memory entries exceed capacity")
            ages = []
            old_timestamps = []
            metadata_field = "last_seen" if payload["version"] == 1 else "age"
            for item in entries:
                if not isinstance(item, dict) or set(item) != {"descriptor", "strength", metadata_field}:
                    raise ValueError("invalid memory entry")
                values = item["descriptor"]
                if not isinstance(values, list) or len(values) != _DESCRIPTOR_SIZE:
                    raise ValueError("invalid memory descriptor")
                if not all(_finite_number(value) for value in values):
                    raise ValueError("memory descriptor must be finite")
                descriptor = np.asarray(values, dtype=np.float64)
                norm = float(np.linalg.norm(descriptor))
                if not math.isfinite(norm) or norm <= 0.0:
                    raise ValueError("memory descriptor must have finite nonzero magnitude")
                strength = item["strength"]
                metadata = item[metadata_field]
                if (not _finite_number(strength) or not 0.0 < strength <= _MAX_STRENGTH
                        or not _finite_number(metadata) or metadata < 0.0):
                    raise ValueError("invalid memory entry metadata")
                if payload["version"] == 1:
                    old_timestamps.append(float(metadata))
                else:
                    ages.append(float(metadata))
                memory._entries.append(
                    FamiliarityEntry(descriptor / norm, float(strength), 0.0)
                )
            if payload["version"] == 1:
                latest = max(old_timestamps, default=0.0)
                ages = [latest - timestamp for timestamp in old_timestamps]
            if not all(math.isfinite(age) for age in ages):
                raise ValueError("invalid memory ages")
            memory._saved_ages = ages
        except (TypeError, OverflowError) as exc:
            raise ValueError("invalid memory data") from exc
        return memory


class ShortTermMemory:
    """Transient scalar target and threat activities for one episode."""

    def __init__(self, target_decay_rate: float = 1.0, threat_decay_rate: float = 1.0):
        for rate in (target_decay_rate, threat_decay_rate):
            if not _finite_number(rate) or rate <= 0.0:
                raise ValueError("decay rates must be finite and positive")
        self.target_decay_rate = float(target_decay_rate)
        self.threat_decay_rate = float(threat_decay_rate)
        self.reset()

    def update(self, target: float, threat: float, dt: float) -> tuple[float, float]:
        if (not _finite_number(target) or not 0.0 <= target <= 1.0
                or not _finite_number(threat) or not 0.0 <= threat <= 1.0
                or not _finite_number(dt) or dt <= 0.0):
            raise ValueError("target and threat must be in [0, 1], dt finite and positive")
        target_retention = math.exp(-self.target_decay_rate * dt)
        threat_retention = math.exp(-self.threat_decay_rate * dt)
        self.target = self.target * target_retention + float(target) * (1.0 - target_retention)
        self.threat = self.threat * threat_retention + float(threat) * (1.0 - threat_retention)
        return self.target, self.threat

    def reset(self) -> None:
        self.target = 0.0
        self.threat = 0.0


def _finite_number(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")
