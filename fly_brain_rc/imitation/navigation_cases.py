"""Deterministic first-stage navigation challenges, separate from selection suite.

Cylinder layouts test multiple obstacles and initially invisible goals, not
room exploration. Split-specific RNG keeps training and final maps distinct.
"""
import random
from ..sim.scenarios import Scenario, BLUE, TEAL


def navigation_cases(seed, split):
    if split not in ('train', 'selection', 'final'):
        raise ValueError('train, selection or final split required')
    rng = random.Random(f'navigation-v1:{split}:{seed}')
    cases = []
    for family in ('double', 'staggered', 'search'):
        for side in (-1, 1):
            dx, dy = rng.uniform(-.06, .06), rng.uniform(-.08, .08)
            if family == 'double':
                points = [(1., side*.15), (1.65, -side*.45), (1.4, side*1.35)]
                goal, pose = (2.5+dx, side*.1+dy), (0., 0., 0.)
            elif family == 'staggered':
                points = [(.85, side*.4), (1.55, -side*.4), (2.2, side*.6)]
                goal, pose = (2.65+dx, -side*.25+dy), (0., 0., 0.)
            else:
                points = [(1.1, side*.35), (1.7, -side*1.1), (.8, side*1.5)]
                goal, pose = (2.4+dx, -side*.1+dy), (0., 0., side*rng.uniform(1.9, 2.3))
            obstacles = tuple(((x+dx, y+dy), BLUE if i%2 == 0 else TEAL)
                              for i, (x, y) in enumerate(points))
            case = Scenario(f'navigation-v1-{split}-{seed}-{family}-{side}', obstacles, goal)
            cases.append((case, pose, family))
    return cases


def sequential_cases(seed, split, *, count=8, obstacles=2):
    """Independent per-obstacle geometry; gentle, two, then three obstacles.

    Open sides leave bypass room. These are cylinder-navigation challenges,
    not rooms or a guarantee of teacher solvability.
    """
    if split not in ('train', 'selection', 'final'):
        raise ValueError('train, selection or final split required')
    if type(count) is not int or not 1 <= count <= 100 or obstacles not in (1, 2, 3):
        raise ValueError('bounded count and one to three obstacles required')
    rng = random.Random(f'sequential-v1:{split}:{seed}:{obstacles}')
    cases = []
    for index in range(count):
        side = -1 if index % 2 else 1
        points = [(rng.uniform(.85, 1.05), side*rng.uniform(.16, .40))]
        if obstacles >= 2:
            points.append((rng.uniform(1.58, 1.80), -side*rng.uniform(.30, .65)))
        if obstacles == 3:
            # A third visible object farther to the side, retaining a roomy route.
            points.append((rng.uniform(1.05, 1.65), side*rng.uniform(1.35, 1.60)))
        goal = (rng.uniform(2.55, 2.68), side*rng.uniform(.05, .23))
        pose = (0., rng.uniform(-.06, .06), rng.uniform(-.12, .12))
        objects = tuple((point, BLUE if i % 2 == 0 else TEAL)
                        for i, point in enumerate(points))
        cases.append((Scenario(f'sequential-v1-{split}-{seed}-{obstacles}-{index}', objects, goal),
                      pose, 'sequential'))
    return cases


def mixed_cases(seed, split, stage):
    """Explicit simple/complex mixtures: 8:4, 6:6, 4:8 per stage.

    Simple includes existing straight, cylinder, turn and slalom skills;
    complex adds independently varied sequential obstacles, not room search.
    """
    if split not in ('train','selection','final') or type(stage) is not int or stage not in (0,1,2):
        raise ValueError('known split and stage 0..2 required')
    from .curriculum import training_case
    rng = random.Random(f'mixed-v1:{split}:{seed}:{stage}')
    simple_count = (8,6,4)[stage]
    result = []
    for i in range(simple_count):
        case, pose, _ = training_case(rng.randrange(1, 1000000000), (72,78,84,90)[i%4])
        result.append((Scenario(f'mixed-v1-{split}-{seed}-{stage}-simple-{i}',case.obstacles,case.goal),pose,'simple'))
    for case, pose, _ in sequential_cases(rng.randrange(1,1000000000),split,
                                          count=12-simple_count,obstacles=2 if stage==0 else 3):
        result.append((Scenario(f'mixed-v1-{split}-{seed}-{stage}-complex-{len(result)}',case.obstacles,case.goal),pose,'complex'))
    return result
