"""Deterministic, separately named training and held-out evaluation maps."""
from dataclasses import dataclass
import math
import random
from .scenarios import Scenario, BLUE


@dataclass(frozen=True)
class SuiteCase:
    identifier: str
    seed: int
    scenario: Scenario
    start_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)


def recovery_cases(seed, count):
    """Near-contact starts, alternating front/rear cylinder and wall."""
    if type(count) is not int or count < 0:
        raise ValueError('count must be nonnegative')
    layouts = (
        ('recover_front_cylinder', (((.51, 0), BLUE),), (-1.8, .6), (0,0,0)),
        ('recover_rear_cylinder', (((-.33, 0), BLUE),), (1.8, .6), (0,0,0)),
        ('recover_front_wall', (), (0,.6), (2.77,0,0)),
        ('recover_rear_wall', (), (0,.6), (-2.95,0,0)),
    )
    return tuple(SuiteCase(f'{layouts[i%4][0]}-{seed}-{i}', seed+i,
                           Scenario(f'{layouts[i%4][0]}-{seed}-{i}', layouts[i%4][1], layouts[i%4][2]),
                           layouts[i%4][3]) for i in range(count))


def training_cases(seed, count, *, occluded=False):
    if type(count) is not int or count < 0:
        raise ValueError('count must be nonnegative')
    rng = random.Random(seed)
    result = []
    for index in range(count):
        angle = rng.uniform(-math.pi, math.pi)
        radius = rng.uniform(1.8, 2.6)
        name = f'{"trainb" if occluded else "train"}-{seed}-{index:03d}'
        obstacles = ()
        if occluded:
            along = rng.uniform(.9, 1.3)
            lateral = rng.choice((-1, 1)) * rng.uniform(.25, .45)
            obstacles = (((along * math.cos(angle) - lateral * math.sin(angle),
                           along * math.sin(angle) + lateral * math.cos(angle)), BLUE),)
        result.append(SuiteCase(name, seed + index, Scenario(name, obstacles, (radius * math.cos(angle), radius * math.sin(angle)))))
    return tuple(result)


def type_a_cases():
    return tuple(SuiteCase(f'type-a-{i:02d}', 10000 + i,
                 Scenario(f'type-a-{i:02d}', (), (-2.0, (i - 4.5) * .22))) for i in range(10))


def type_b_cases():
    return tuple(SuiteCase(f'type-b-{i:02d}', 20000 + i,
                 Scenario(f'type-b-{i:02d}', (((1.1, .12 * (-1 if i % 2 else 1)), BLUE),),
                          (2.3, (-1 if i % 2 else 1) * (.65 + .06 * i)))) for i in range(10))
