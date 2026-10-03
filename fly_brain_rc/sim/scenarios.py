from dataclasses import dataclass


Color = tuple[float, float, float]
Obstacle = tuple[tuple[float, float], Color]

BLUE: Color = (0.25, 0.60, 0.88)
TEAL: Color = (0.22, 0.72, 0.72)


@dataclass(frozen=True)
class Scenario:
    name: str
    obstacles: tuple[Obstacle, ...]
    goal: tuple[float, float]


def _obstacles(points: list[tuple[float, float]]) -> tuple[Obstacle, ...]:
    return tuple((point, BLUE if index % 2 == 0 else TEAL) for index, point in enumerate(points))


_SCENARIOS = {
    "single_cylinder": Scenario("single_cylinder",_obstacles([(1.15,.18)]),(2.35,0.0)),
    "corridor": Scenario(
        "corridor",
        _obstacles(
            [
                (0.45, -1.00), (0.45, 1.00),
                (1.05, -1.00), (1.05, 1.00),
                (1.65, -1.00), (1.65, 1.00),
                (2.25, -1.00), (2.25, 1.00),
            ]
        ),
        (2.45, 0.0),
    ),
    "turn_corridor": Scenario(
        "turn_corridor",
        _obstacles(
            [
                (0.55, -1.05), (0.55, 1.05),
                (1.15, -1.05), (1.15, 1.45),
                (1.75, -1.05), (2.35, -0.45),
                (1.75, 1.45), (2.35, 1.45),
            ]
        ),
        (2.30, 0.85),
    ),
    "slalom": Scenario(
        "slalom",
        _obstacles(
            [
                (0.65, 0.58), (1.25, -0.58), (1.85, 0.58), (2.35, -0.58),
                (0.65, -1.45), (1.25, 1.45), (1.85, -1.45), (2.35, 1.45),
            ]
        ),
        (2.55, 0.0),
    ),
    "blocked_goal": Scenario(
        "blocked_goal",
        _obstacles(
            [
                (1.15, 0.18), (0.75, -1.20), (0.75, 1.20),
                (1.55, -1.20), (1.55, 1.20), (2.25, -1.20), (2.25, 1.20),
            ]
        ),
        (2.35, 0.0),
    ),
}


def scenario_names() -> list[str]:
    return sorted(_SCENARIOS)


def get_scenario(name: str) -> Scenario:
    try:
        return _SCENARIOS[name]
    except KeyError as error:
        raise ValueError(f"unknown scenario: {name}") from error
