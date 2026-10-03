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
import os
import tempfile

from .contracts import ActuatorCommand, BrainTelemetry, ObjectObservation, SensorFrame
from .recovery import RecoveryManoeuvre
from .navigation_memory import NavigationMemory


DEFAULT_ACTIONS = (
    (0.30, -0.65), (0.30, 0.0), (0.30, 0.65),
    (0.70, -0.75), (0.70, -0.35), (0.70, 0.0), (0.70, 0.35), (0.70, 0.75),
    (-0.70, -0.80), (-0.70, 0.0), (-0.70, 0.80),
)
ACTION_DURATION = 0.2
POLICY_VERSION = 7


def _bearing(label: str, observation: ObjectObservation | None) -> str:
    if observation is None:
        return f"{label}_none"
    if observation.bearing < -0.18:
        return f"{label}_left"
    if observation.bearing > 0.18:
        return f"{label}_right"
    return f"{label}_center"


def observation_state(sensor: SensorFrame, target: ObjectObservation | None,
                      obstacle: ObjectObservation | None, *, previous: str = 'none',
                      failed: str = 'none', blocked_action: int | None = None) -> tuple[str, ...]:
    """Compact policy state containing only camera and embodied readings."""
    obstacle_state = "clear" if obstacle is None else _bearing("obstacle", obstacle)
    def range_bin(value):
        return "blocked" if value < .25 else "near" if value < .60 else "clear"
    distance = range_bin(sensor.tof)
    movement = "reverse" if sensor.speed < -.025 else "forward" if sensor.speed > .025 else "stalled"
    contact = ("contact_left" if sensor.contact_left else "contact_right" if sensor.contact_right
               else "contact_center" if sensor.contact else "no_contact")
    obstacle_size = "large" if obstacle is not None and obstacle.apparent_area >= .08 else "small"
    longitudinal = ('contact_front' if sensor.contact_front else 'contact_rear' if sensor.contact_rear
                    else 'contact_unknown' if sensor.contact else 'no_contact')
    return (_bearing("target", target), obstacle_state, obstacle_size,
            range_bin(sensor.range_left), distance, range_bin(sensor.range_right), movement, contact,
            longitudinal)


class ExplorationReward:
    """Intrinsic drive: seek unvisited cells and avoid behavioural freezing."""

    def __init__(self, *, cell_size: float = 0.45, new_cell_reward: float = 1.0,
                 idle_penalty: float = 0.01, max_idle_penalty: float = 0.12,
                 contact_penalty: float = 1.5,
                 goal_reward: float = 50.0):
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.cell_size = cell_size
        self.new_cell_reward = new_cell_reward
        self.idle_penalty = idle_penalty
        self.max_idle_penalty = max_idle_penalty
        self.contact_penalty = contact_penalty
        self.goal_reward = goal_reward
        self.reset((0.0, 0.0))

    def _cell(self, position: tuple[float, float]) -> tuple[int, int]:
        return tuple(int(value // self.cell_size) for value in position)

    def reset(self, position: tuple[float, float]) -> None:
        self.visited = {self._cell(position)}
        self.stationary_steps = 0
        self._contact_active = False
        self._target_potential = None
        self._reverse_seconds = 0.0

    def step(self, position: tuple[float, float], *, moved: bool, contact: bool, reached: bool,
             target: ObjectObservation | None = None, speed: float = 0.0, dt: float = .02) -> float:
        reward = 0.0
        cell = self._cell(position)
        if cell not in self.visited:
            self.visited.add(cell)
            reward += self.new_cell_reward
        if moved:
            self.stationary_steps = 0
        else:
            self.stationary_steps += 1
            reward -= min(self.max_idle_penalty, self.idle_penalty * self.stationary_steps)
        if contact and not self._contact_active:
            reward -= self.contact_penalty
        self._contact_active = contact
        if target is None:
            self._target_potential = None
        else:
            # Reward improvement, not merely having the ball in the frame.
            potential = 4.0 * target.apparent_area - 0.15 * abs(target.bearing)
            if self._target_potential is not None:
                reward += max(-0.10, min(0.10, 2.0 * (potential - self._target_potential)))
            self._target_potential = potential
        self._reverse_seconds = self._reverse_seconds + dt if speed < -.025 else 0.0
        if self._reverse_seconds > .6:
            # Allow a short retreat, then increase the cost to 0.4 per second.
            rate = min(.4, .2 + .1 * (self._reverse_seconds - .6))
            reward -= rate * dt
        if reached:
            reward += self.goal_reward
        return reward

@dataclass
class QLearningAgent:
    actions: tuple[tuple[float, float], ...] = DEFAULT_ACTIONS
    alpha: float = 0.18
    gamma: float = 0.98
    epsilon: float = 0.20
    seed: int = 0

    def __post_init__(self):
        if not self.actions or not 0 < self.alpha <= 1 or not 0 <= self.gamma <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("invalid Q-learning settings")
        self.actions = tuple((float(traction), float(steering)) for traction, steering in self.actions)
        self._rng = random.Random(self.seed)
        self.table: dict[tuple[str, ...], list[float]] = {}
        self.training_runs = 0
        self.task = 'navigation'
        self.sensor_schema = 1
        self.legacy_inference = False

    def values(self, state):
        state = tuple(state)
        candidates = [state]
        if state[0].startswith('target_memory_'):
            candidates.append((state[0].replace('target_memory_','target_',1),)+state[1:])
        if self.task == 'bypass-v1' and len(state)==10:
            candidates += [key[:9] for key in candidates]
        for key in candidates:
            if key in self.table:
                return self.table[key]
        return [0.0] * len(self.actions)

    def _values(self, state: tuple[str, ...]) -> list[float]:
        return self.table.setdefault(tuple(state), list(self.values(state)))

    def value(self, state: tuple[str, ...], action: int) -> float:
        return self._values(state)[action]

    def admissible_actions(self, state):
        # Geometry contacts act as directional bumpers; failed travel also stays
        # inhibited until encoder progress, even after a contact flickers off.
        failed = state[10] if len(state) > 10 else 'none'
        bumper = state[8] if len(state) > 8 else 'no_contact'
        blocked_action = int(state[11]) if len(state) > 11 and state[11] != 'none' else None
        admissible = [i for i,(t,_) in enumerate(self.actions)
                      if not (t > 0 and (bumper == 'contact_front' or failed == 'forward_failed'))
                      and not (t < 0 and (bumper == 'contact_rear' or failed == 'reverse_failed'))]
        if not admissible:
            # Conflicting stale failure and current bumper: the bumper wins.
            permitted = [i for i,(t,_) in enumerate(self.actions)
                         if (t <= 0 if bumper == 'contact_front' else t >= 0)]
            def turn_category(steer):
                return -1 if steer < -.1 else 1 if steer > .1 else 0
            blocked_turn = None if blocked_action is None else turn_category(self.actions[blocked_action][1])
            admissible = [i for i in permitted if turn_category(self.actions[i][1]) != blocked_turn]
            admissible = admissible or permitted or list(range(len(self.actions)))
        return admissible

    def choose(self, state: tuple[str, ...], *, explore: bool = True) -> int:
        admissible = self.admissible_actions(state)
        failed = state[10] if len(state) > 10 else 'none'
        bumper = state[8] if len(state) > 8 else 'no_contact'
        if explore and self._rng.random() < self.epsilon:
            return self._rng.choice(admissible)
        values = self.values(state)
        best = max(values[i] for i in admissible)
        candidates = [i for i in admissible if values[i] == best]
        # Deterministic, low-speed fallback toward a visible target or a free
        # side. With no target, keep a gentle search turn, not blind reversing.
        target_direction = state[0].replace('target_memory_','target_',1)
        steer = -0.65 if target_direction == 'target_left' else .65 if target_direction == 'target_right' else 0.0
        if state[4] == 'blocked' or state[1] == 'obstacle_center':
            rank = {'blocked': 0, 'near': 1, 'clear': 2}
            steer = -.65 if rank[state[3]] > rank[state[5]] else .65
        elif state[0] == 'target_none':
            steer = -.65 if state[1] == 'obstacle_right' else .65
        preferred_traction = -.7 if bumper == 'contact_front' or failed == 'forward_failed' else .3
        return min(candidates, key=lambda i: (abs(self.actions[i][0]-preferred_traction),
                                             abs(self.actions[i][1]-steer), i))

    def update(self, state: tuple[str, ...], action: int, reward: float,
               next_state: tuple[str, ...], *, terminal: bool) -> None:
        values = self._values(state)
        future = 0.0 if terminal else max(self._values(next_state)[i] for i in self.admissible_actions(next_state))
        values[action] += self.alpha * (reward + self.gamma * future - values[action])

    def save(self, path: str | Path) -> None:
        payload = {
            "version": POLICY_VERSION, "actions": self.actions, "alpha": self.alpha,
            "gamma": self.gamma, "epsilon": self.epsilon,
            "training_runs": self.training_runs,
            "task": self.task,
            "sensor_schema": self.sensor_schema,
            "table": {"|".join(state): values for state, values in self.table.items()},
        }
        path = Path(path)
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as handle:
            temporary = handle.name
            try:
                json.dump(payload,handle,sort_keys=True,allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                os.unlink(temporary)
                raise
        try:
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path, *, epsilon: float = 0.0, seed: int = 0) -> "QLearningAgent":
        payload = json.loads(Path(path).read_text())
        if payload.get("version") not in (4, 5, 6, POLICY_VERSION):
            raise ValueError("unsupported RL policy; retrain with the current reward model")
        agent = cls(tuple(tuple(action) for action in payload["actions"]),
                    alpha=float(payload["alpha"]), gamma=float(payload["gamma"]), epsilon=epsilon, seed=seed)
        grouped = {}
        ordinary = {}
        for key, values in payload['table'].items():
            state = tuple(key.split('|'))
            if payload.get('version') == 5:
                state = state[:9]
            elif payload.get('version') == 4:
                state = state[:8] + ('no_contact' if state[7] == 'no_contact' else 'contact_unknown',)
            grouped.setdefault(state, []).append(list(map(float, values)))
            if payload.get('version') == 5 and key.split('|')[10] == 'none':
                ordinary.setdefault(state, []).append(list(map(float,values)))
        for state, rows in grouped.items():
            # No visit counts exist in v5. Nonzero-only averaging is a warm
            # start, not an exact statistical reconstruction of experience.
            columns = list(zip(*rows))
            preferred = list(zip(*ordinary.get(state,rows)))
            estimates = []
            for column, normal in zip(columns,preferred):
                known = [v for v in normal if v != 0] or [v for v in column if v != 0]
                estimates.append(sum(known)/len(known) if known else 0.0)
            agent.table[state] = estimates
        agent.training_runs = int(payload.get("training_runs", 0))
        agent.task = payload.get('task','navigation')
        agent.sensor_schema = int(payload.get('sensor_schema',1))
        if agent.task not in ('navigation','bypass-v1'):
            raise ValueError('unsupported policy task')
        agent.legacy_inference = False
        if any(len(values) != len(agent.actions) for values in agent.table.values()):
            raise ValueError("invalid RL policy table")
        return agent


class RLController:
    """Shared encoder-based manoeuvre memory for training and inference."""

    def __init__(self, agent):
        self.agent = agent
        self.action = None
        self.blocked_seconds = 0.0
        self.timestamp = None
        self.recovery = RecoveryManoeuvre()
        self.navigation = NavigationMemory(enable_bypass=agent.task != 'bypass-v1')
        self.sensor = None
        self.overriding = False

    @property
    def mode(self):
        return self.recovery.phase if self.recovery.phase != 'drive' else 'bypass_'+self.navigation.phase if self.navigation.phase != 'drive' else 'drive'

    def observe(self, sensor, target, obstacle):
        self.sensor = sensor
        if self.agent.task == 'bypass-v1':
            self.navigation.observe(sensor,target,obstacle)
            return self._policy_state(sensor,target,obstacle)
        traction = None if self.action is None else self.agent.actions[self.action][0]
        old_phase = self.recovery.phase
        self.recovery.observe(sensor, traction, obstacle, target)
        self.navigation.observe(sensor,target,obstacle,recovering=self.recovery.phase != 'drive',
                                recovered=old_phase != 'drive' and self.recovery.phase == 'drive')
        if old_phase == 'drive' and self.recovery.phase == 'retreat' and self.navigation.phase != 'drive':
            self.recovery.turn = self.navigation.side
        elapsed = 0 if self.timestamp is None else max(0, sensor.timestamp-self.timestamp)
        self.timestamp = sensor.timestamp
        if traction is not None and sensor.contact and abs(sensor.encoder_delta) < .0001:
            self.blocked_seconds += elapsed
            if self.blocked_seconds >= .4 - 1e-9:
                self.recovery.turn *= -1
                self.recovery._phase('arc' if traction < 0 else 'retreat',sensor.timestamp)
                self.blocked_seconds = 0.0
        else:
            self.blocked_seconds = 0.0
        return self._policy_state(sensor,target,obstacle)

    def _policy_state(self, sensor, target, obstacle):
        effective = target if target is not None else self.navigation.recalled_target(sensor)
        state = observation_state(sensor,effective,obstacle)
        if target is None and effective is not None:
            state = (state[0].replace('target_','target_memory_',1),)+state[1:]
        if self.agent.task == 'bypass-v1':
            left,right = sensor.range_side_left < .25,sensor.range_side_right < .25
            side = 'both' if left and right else 'left' if left else 'right' if right else 'clear'
            rear = 'near' if sensor.range_rear < .35 else 'clear'
            state += (f'side_{side}:rear_{rear}',)
        return state

    def choose(self, state, *, explore):
        command = self.recovery.command(self.sensor)
        if command is None:
            command = self.navigation.command()
        self.overriding = command is not None
        if self.overriding:
            t,w = command
            action = min(range(len(self.agent.actions)),key=lambda i: abs(self.agent.actions[i][0]-t)*3+abs(self.agent.actions[i][1]-w))
        else:
            action = self.agent.choose(state, explore=explore)
        if self.action is not None and self.agent.actions[action][0]*self.agent.actions[self.action][0] < 0:
            self.blocked_seconds = 0.0
        self.action = action
        return action


class QLearningBrain:
    """Frozen trained policy behind the regular runtime brain interface."""

    def __init__(self, agent: QLearningAgent):
        self.agent = agent
        self.reset()

    def reset(self):
        self.controller = RLController(self.agent)
        self._action = None
        self._until = -float('inf')

    def step(self, sensor: SensorFrame, target: ObjectObservation | None, motion,
             familiarity: float, dt: float, *, obstacle: ObjectObservation | None = None):
        old_mode = self.controller.mode
        state = self.controller.observe(sensor, target, obstacle)
        unsafe = self._action is not None and self._action not in self.agent.admissible_actions(state)
        if self._action is None or unsafe or sensor.timestamp >= self._until - 1e-9 or old_mode != self.controller.mode:
            self._action = self.controller.choose(state, explore=False)
            self._until = sensor.timestamp + ACTION_DURATION
        action = self._action
        traction, steering = self.agent.actions[action]
        value = self.agent.values(state)[action]
        command = ActuatorCommand(traction, steering, sensor.timestamp, 0)
        telemetry = BrainTelemetry(
            populations={"rl_best_value": value, "rl_known_states": float(len(self.agent.table)),
                         "rl_target_memory": float(self.controller.navigation.vector is not None),
                         "rl_memory_confidence": self.controller.navigation.confidence,
                         "rl_bypass_active": float(self.controller.navigation.phase != 'drive')},
            familiarity=familiarity,
            threat=float(sensor.contact),
            motor_left=max(0.0, -steering), motor_right=max(0.0, steering),
            motor_forward=max(0.0, traction), motor_reverse=max(0.0, -traction),
            traction=traction, steering=steering,
        )
        return command, telemetry
