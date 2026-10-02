"""Sensor-only fixed-topology rate brain and continuous arrival criteria."""

import math

import numpy as np

from fly_brain_rc.contracts import ActuatorCommand, BrainTelemetry
from fly_brain_rc.neural.compass import CompassState
from fly_brain_rc.neural.dynamics import RateLayer
from fly_brain_rc.neural.genome import FlyGenome


class NeuralFlyBrain:
    POPULATIONS = ('target_left', 'target_center', 'target_right', 'target_size',
                   'obstacle_left', 'obstacle_center', 'obstacle_right', 'obstacle_size',
                   'flow_left', 'flow_center', 'flow_right', 'looming', 'proximity',
                   'contact', 'familiarity', 'habituation', 'exploration', 'threat',
                   'motor_left', 'motor_right', 'motor_forward', 'motor_reverse')

    def __init__(self, genome=None):
        self.genome = genome if genome is not None else FlyGenome.default()
        if not isinstance(self.genome, FlyGenome):
            raise ValueError('genome must be a FlyGenome')
        self.compass = CompassState()
        self.reset()

    @property
    def target_memory(self):
        return self._target_memory.copy()

    def reset(self):
        self.compass.reset()
        self._state = np.zeros(len(self.POPULATIONS))
        self._target_memory = np.zeros(self.compass.RING_SIZE)
        self._visits = np.zeros(self.compass.RING_SIZE)
        self._sequence = 0

    def step(self, sensor, target, motion, familiarity, dt, *, obstacle=None):
        values = (sensor.yaw_rate, sensor.encoder_delta, sensor.tof, sensor.timestamp,
                  motion.flow_left, motion.flow_center, motion.flow_right,
                  motion.optic_rotation, motion.looming, familiarity, dt)
        if not all(math.isfinite(value) for value in values) or dt <= 0 or not 0 <= familiarity <= 1:
            raise ValueError('brain inputs must be finite, familiarity in [0, 1], dt positive')
        if target is not None and (not all(math.isfinite(value) for value in
                                         (target.bearing, target.apparent_area, target.confidence))
                                   or not 0 <= target.apparent_area <= 1
                                   or not 0 <= target.confidence <= 1):
            raise ValueError('invalid target observation')
        if obstacle is not None and (obstacle.class_name != 'obstacle'
                                     or not all(math.isfinite(value) for value in
                                                (obstacle.bearing, obstacle.apparent_area,
                                                 obstacle.confidence))
                                     or not 0 <= obstacle.apparent_area <= 1
                                     or not 0 <= obstacle.confidence <= 1):
            raise ValueError('invalid obstacle observation')
        g = self.genome
        old = dict(zip(self.POPULATIONS, self._state))
        heading = self.compass.heading
        angles = np.linspace(-math.pi, math.pi, self.compass.RING_SIZE, endpoint=False)
        relative = np.arctan2(np.sin(angles - heading), np.cos(angles - heading))
        memory_left = float(np.sum(self._target_memory * np.maximum(-np.sin(relative), 0)))
        memory_right = float(np.sum(self._target_memory * np.maximum(np.sin(relative), 0)))
        confidence = target.confidence if target is not None else 0.0
        bearing = target.bearing if target is not None else 0.0
        left = confidence * max(0.0, -math.sin(bearing))
        right = confidence * max(0.0, math.sin(bearing))
        center = confidence * max(0.0, math.cos(bearing))
        area = target.apparent_area if target is not None else 0.0
        obstacle_confidence = obstacle.confidence if obstacle is not None else 0.0
        obstacle_bearing = obstacle.bearing if obstacle is not None else 0.0
        obstacle_area = obstacle.apparent_area if obstacle is not None else 0.0
        obstacle_left = obstacle_confidence * max(0.0, -math.sin(obstacle_bearing))
        obstacle_right = obstacle_confidence * max(0.0, math.sin(obstacle_bearing))
        obstacle_center = obstacle_confidence * max(0.0, math.cos(obstacle_bearing))
        obstacle_size = min(1.0, obstacle_area / 0.02)
        size = min(1.0, area / g.arrival_area)
        proximity = float(np.clip(1 - sensor.tof / g.proximity_distance, 0, 1))
        looming = g.looming_gain * float(np.clip(motion.looming, 0, 1))
        # Seeing an obstacle is a navigation cue: steer around it and ease
        # off the throttle. Actual retreat is reserved for a close range,
        # optical expansion, or a physical contact.
        threat_input = max(looming, proximity, float(sensor.contact))
        sensory_threat = max(old['looming'], old['proximity'], old['contact'])
        displacement = math.tanh(math.hypot(*self.compass.home_vector))
        return_familiarity = (1 - displacement) * old['habituation']
        novelty = float(np.sum(self.compass.activity * (1 - self._visits)))
        # All recurrent drives read the same snapshot. Sensory feed-through
        # permits immediate threat response without sequential population updates.
        drive = {name: -4.0 for name in self.POPULATIONS}
        for name, value in (('target_left', left), ('target_center', center), ('target_right', right),
                            ('target_size', size), ('obstacle_left', obstacle_left),
                            ('obstacle_center', obstacle_center), ('obstacle_right', obstacle_right),
                            ('obstacle_size', obstacle_size), ('flow_left', abs(motion.flow_left)),
                            ('flow_center', abs(motion.flow_center)), ('flow_right', abs(motion.flow_right)),
                            ('looming', looming), ('proximity', proximity), ('contact', float(sensor.contact)),
                            ('familiarity', familiarity)):
            drive[name] += g.sensory_gain * float(np.clip(value, 0, 1))
        drive['habituation'] += 6 * familiarity + old['familiarity'] + 2 * old['habituation']
        drive['exploration'] += g.exploration_gain * (1 - familiarity + novelty + displacement) - 4 * old['threat'] - 2 * old['habituation'] - old['familiarity'] - return_familiarity
        drive['threat'] += g.threat_gain * threat_input + 2 * (old['threat'] + sensory_threat)
        # Slow asymmetry and directional visitation break exploration ties.
        turn = 0.15 + float(np.sum((1 - self._visits) * np.sin(relative)) / len(relative))
        avoidance = math.tanh(motion.flow_left - motion.flow_right + 0.15)
        exploration = old['exploration'] * (1 - confidence)
        steering_drive = g.target_gain * (right - left + memory_right - memory_left +
                                          0.5 * (old['target_right'] - old['target_left'])) + g.exploration_turn_gain * exploration * turn + (threat_input + sensory_threat) * avoidance + 0.5 * (old['flow_left'] - old['flow_right'])
        obstacle_turn = (obstacle_right - obstacle_left
                         + .75 * obstacle_center * turn
                         + .5 * (old['obstacle_right'] - old['obstacle_left']))
        steering_drive -= g.threat_gain * obstacle_turn
        steering_drive -= g.threat_gain * (float(sensor.contact_left) - float(sensor.contact_right))
        drive['motor_right'] = -2 + steering_drive - 2 * old['motor_left']
        drive['motor_left'] = -2 - steering_drive - 2 * old['motor_right']
        braking = confidence * min(2.0, area / g.arrival_area)
        sensory_approach = old['target_center'] - old['target_size']
        drive['motor_forward'] = -2 + 4 * confidence + sensory_approach + g.exploration_gain * exploration - 4 * braking - 2 * obstacle_size - g.threat_gain * threat_input - 2 * (old['threat'] + sensory_threat) - old['flow_center']
        drive['motor_reverse'] = -4 + 2 * braking + g.threat_gain * threat_input + 2 * (old['threat'] + sensory_threat)
        tau = np.full(len(self.POPULATIONS), g.tau)
        tau[self.POPULATIONS.index('habituation')] = 1 / g.habituation_rate
        next_state = RateLayer(tau=tau).step(self._state, np.zeros((len(tau), len(tau))),
                                            np.array([drive[name] for name in self.POPULATIONS]), dt)
        retention = math.exp(-g.memory_decay * dt)
        next_memory = self._target_memory * retention
        if target is not None:
            bump = np.exp(8 * (np.cos(angles - heading - bearing) - 1))
            next_memory += (1 - retention) * confidence * bump / bump.sum()
        next_visits = np.clip(self._visits * math.exp(-g.visit_decay * dt) +
                              dt * self.compass.activity, 0, 1)
        self.compass.update(sensor.yaw_rate, motion.optic_rotation, sensor.encoder_delta, dt)
        self._state, self._target_memory, self._visits = next_state, next_memory, next_visits
        self._sequence += 1
        populations = dict(zip(self.POPULATIONS, map(float, next_state)))
        traction = populations['motor_forward'] - populations['motor_reverse']
        steering = populations['motor_right'] - populations['motor_left']
        command = ActuatorCommand(traction, steering, sensor.timestamp, self._sequence)
        telemetry = BrainTelemetry(populations=populations, compass_ring=tuple(self.compass.activity),
            target_memory=tuple(next_memory), path_integration=tuple(self.compass.home_vector),
            familiarity=familiarity, habituation=populations['habituation'], threat=populations['threat'],
            motor_left=populations['motor_left'], motor_right=populations['motor_right'],
            motor_forward=populations['motor_forward'], motor_reverse=populations['motor_reverse'],
            traction=traction, steering=steering)
        return command, telemetry


class ArrivalTracker:
    def __init__(self, bearing_threshold=0.12, area_min=0.10, area_max=0.22,
                 traction_threshold=0.05, speed_threshold=0.05, safe_tof=0.20, duration=3.0):
        values = (bearing_threshold, area_min, area_max, traction_threshold, speed_threshold, safe_tof, duration)
        if not all(math.isfinite(value) and value > 0 for value in values) or not area_min < area_max <= 1:
            raise ValueError('invalid arrival thresholds')
        self.bearing_threshold, self.area_min, self.area_max = bearing_threshold, area_min, area_max
        self.traction_threshold, self.speed_threshold = traction_threshold, speed_threshold
        self.safe_tof, self.duration = safe_tof, duration
        self.reset()

    def reset(self):
        self.elapsed = 0.0
        self.last_reason = 'waiting_for_target'

    @property
    def arrived(self):
        return self.elapsed >= self.duration - 1e-12

    def update(self, target, command, speed, tof, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        stable = (target is not None and target.confidence > 0
                  and all(math.isfinite(value) for value in (target.bearing, target.apparent_area,
                           target.confidence, command.traction, speed, tof))
                  and abs(target.bearing) <= self.bearing_threshold
                  and self.area_min <= target.apparent_area <= self.area_max
                  and abs(command.traction) <= self.traction_threshold
                  and abs(speed) <= self.speed_threshold and tof >= self.safe_tof)
        if stable:
            self.elapsed = min(self.duration, self.elapsed + dt)
            self.last_reason = 'stable'
        else:
            # Once settling has begun, tolerate one-frame visual/odometry
            # jitter instead of discarding an otherwise steady stop.
            relaxed = (target is not None and target.confidence > 0
                       and abs(target.bearing) <= self.bearing_threshold * 1.35
                       and self.area_min * .8 <= target.apparent_area <= self.area_max * 1.2
                       and abs(command.traction) <= self.traction_threshold * 1.5
                       and abs(speed) <= self.speed_threshold * 1.5 and tof >= self.safe_tof)
            self.elapsed = max(0.0, self.elapsed - dt * .25) if self.elapsed and relaxed else 0.0
            self.last_reason = 'visual_or_motion_jitter' if relaxed else 'arrival_condition_lost'
        return self.arrived
