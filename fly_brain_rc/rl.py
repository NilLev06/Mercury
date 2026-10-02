"""Small, dependency-free reinforcement learner for simulated fly driving.

The learner observes only coarse perception categories and chooses one of a
small set of bicycle commands.  World position is used solely by the trainer
to assign exploration reward; it is never part of the learned policy state.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

from .contracts import ActuatorCommand, BrainTelemetry, ObjectObservation, SensorFrame


DEFAULT_ACTIONS = (
    (0.70, -0.75), (0.70, -0.35), (0.70, 0.0), (0.70, 0.35), (0.70, 0.75),
    (-0.70, -0.80), (-0.70, 0.0), (-0.70, 0.80),
)


def _bearing(label: str, observation: ObjectObservation | None) -> str:
    if observation is None:
        return f"{label}_none"
    if observation.bearing < -0.18:
        return f"{label}_left"
    if observation.bearing > 0.18:
        return f"{label}_right"
    return f"{label}_center"


def observation_state(sensor: SensorFrame, target: ObjectObservation | None,
                      obstacle: ObjectObservation | None) -> tuple[str, ...]:
    """Compact policy state containing only camera and embodied readings."""
    obstacle_state = "clear" if obstacle is None else _bearing("obstacle", obstacle)
    distance = "near" if sensor.tof < 0.42 else "far"
    movement = "moving" if abs(sensor.encoder_delta) >= 0.001 else "stalled"
    contact = ("contact_left" if sensor.contact_left else "contact_right" if sensor.contact_right
               else "contact_center" if sensor.contact else "no_contact")
    return (_bearing("target", target), obstacle_state, distance, movement, contact)


class ExplorationReward:
    """Intrinsic drive: seek unvisited cells and avoid behavioural freezing."""

    def __init__(self, *, cell_size: float = 0.45, new_cell_reward: float = 1.0,
                 idle_penalty: float = 0.01, max_idle_penalty: float = 0.12,
                 contact_penalty: float = 1.5, escape_reward: float = 0.5,
                 goal_reward: float = 50.0, max_stall_steps: int = 80):
        if cell_size <= 0 or max_stall_steps <= 0:
            raise ValueError("cell_size and max_stall_steps must be positive")
        self.cell_size = cell_size
        self.new_cell_reward = new_cell_reward
        self.idle_penalty = idle_penalty
        self.max_idle_penalty = max_idle_penalty
        self.contact_penalty = contact_penalty
        self.escape_reward = escape_reward
        self.goal_reward = goal_reward
        self.max_stall_steps = max_stall_steps
        self.reset((0.0, 0.0))

    def _cell(self, position: tuple[float, float]) -> tuple[int, int]:
        return tuple(int(value // self.cell_size) for value in position)

    def reset(self, position: tuple[float, float]) -> None:
        self.visited = {self._cell(position)}
        self.stationary_steps = 0
        self._contact_active = False

    def step(self, position: tuple[float, float], *, moved: bool, contact: bool, reached: bool) -> float:
        reward = 0.0
        cell = self._cell(position)
        if cell not in self.visited:
            self.visited.add(cell)
            reward += self.new_cell_reward
        if moved:
            self.stationary_steps = 0
            if self._contact_active and not contact:
                reward += self.escape_reward
        else:
            self.stationary_steps += 1
            reward -= min(self.max_idle_penalty, self.idle_penalty * self.stationary_steps)
        if contact and not self._contact_active:
            reward -= self.contact_penalty
        self._contact_active = contact
        if reached:
            reward += self.goal_reward
        return reward

    @property
    def is_trapped(self) -> bool:
        return self.stationary_steps >= self.max_stall_steps


@dataclass
class QLearningAgent:
    actions: tuple[tuple[float, float], ...] = DEFAULT_ACTIONS
    alpha: float = 0.18
    gamma: float = 0.92
    epsilon: float = 0.20
    seed: int = 0

    def __post_init__(self):
        if not self.actions or not 0 < self.alpha <= 1 or not 0 <= self.gamma <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("invalid Q-learning settings")
        self.actions = tuple((float(traction), float(steering)) for traction, steering in self.actions)
        self._rng = random.Random(self.seed)
        self.table: dict[tuple[str, ...], list[float]] = {}

    def _values(self, state: tuple[str, ...]) -> list[float]:
        return self.table.setdefault(tuple(state), [0.0] * len(self.actions))

    def value(self, state: tuple[str, ...], action: int) -> float:
        return self._values(state)[action]

    def choose(self, state: tuple[str, ...], *, explore: bool = True) -> int:
        if explore and self._rng.random() < self.epsilon:
            return self._rng.randrange(len(self.actions))
        values = self._values(state)
        best = max(values)
        candidates = [index for index, value in enumerate(values) if value == best]
        return self._rng.choice(candidates)

    def update(self, state: tuple[str, ...], action: int, reward: float,
               next_state: tuple[str, ...], *, terminal: bool) -> None:
        values = self._values(state)
        future = 0.0 if terminal else max(self._values(next_state))
        values[action] += self.alpha * (reward + self.gamma * future - values[action])

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 2, "actions": self.actions, "alpha": self.alpha,
            "gamma": self.gamma, "epsilon": self.epsilon,
            "table": {"|".join(state): values for state, values in self.table.items()},
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path, *, epsilon: float = 0.0) -> "QLearningAgent":
        payload = json.loads(Path(path).read_text())
        if payload.get("version") != 2:
            raise ValueError("unsupported RL policy; retrain with the current reward model")
        agent = cls(tuple(tuple(action) for action in payload["actions"]),
                    alpha=float(payload["alpha"]), gamma=float(payload["gamma"]), epsilon=epsilon)
        agent.table = {tuple(key.split("|")): list(map(float, values)) for key, values in payload["table"].items()}
        if any(len(values) != len(agent.actions) for values in agent.table.values()):
            raise ValueError("invalid RL policy table")
        return agent


class QLearningBrain:
    """Frozen trained policy behind the regular runtime brain interface."""

    def __init__(self, agent: QLearningAgent):
        self.agent = agent

    def step(self, sensor: SensorFrame, target: ObjectObservation | None, motion,
             familiarity: float, dt: float, *, obstacle: ObjectObservation | None = None):
        state = observation_state(sensor, target, obstacle)
        action = self.agent.choose(state, explore=False)
        traction, steering = self.agent.actions[action]
        value = self.agent.value(state, action)
        command = ActuatorCommand(traction, steering, sensor.timestamp, 0)
        telemetry = BrainTelemetry(
            populations={"rl_best_value": value, "rl_known_states": float(len(self.agent.table))},
            familiarity=familiarity,
            threat=float(sensor.contact),
            motor_left=max(0.0, -steering), motor_right=max(0.0, steering),
            motor_forward=max(0.0, traction), motor_reverse=max(0.0, -traction),
            traction=traction, steering=steering,
        )
        return command, telemetry
