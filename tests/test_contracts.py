import dataclasses
import unittest

from fly_brain_rc.contracts import (
    ActuatorCommand,
    BrainTelemetry,
    MotionObservation,
    ObjectObservation,
    RuntimeTelemetry,
    SensorFrame,
)


class ContractTests(unittest.TestCase):
    def test_sensor_frame_exposes_only_embodied_sensors(self):
        names = {field.name for field in dataclasses.fields(SensorFrame)}
        self.assertEqual(names, {"rgb", "gray", "yaw_rate", "encoder_delta", "tof", "contact", "timestamp",
                                 "contact_left", "contact_right", "speed", "range_left", "range_right",
                                 "contact_front", "contact_rear"})
        self.assertTrue(names.isdisjoint({"goal", "goal_distance", "x", "y", "pose", "absolute_yaw"}))

    def test_actuator_command_clamps_outputs(self):
        command = ActuatorCommand(traction=3.0, steering=-2.0, timestamp=1.0, sequence=4)
        self.assertEqual(command.traction, 1.0)
        self.assertEqual(command.steering, -1.0)

    def test_all_runtime_contracts_are_frozen(self):
        for contract in (SensorFrame, ObjectObservation, MotionObservation,
                         ActuatorCommand, BrainTelemetry, RuntimeTelemetry):
            with self.subTest(contract=contract.__name__):
                self.assertTrue(contract.__dataclass_params__.frozen)

    def test_brain_facing_contracts_have_no_ground_truth_fields(self):
        forbidden = {"goal", "goal_x", "goal_y", "goal_distance", "pose", "x", "y", "absolute_yaw"}
        for contract in (SensorFrame, ObjectObservation, MotionObservation,
                         ActuatorCommand, BrainTelemetry):
            with self.subTest(contract=contract.__name__):
                self.assertTrue({field.name for field in dataclasses.fields(contract)}.isdisjoint(forbidden))


if __name__ == "__main__":
    unittest.main()
