"""Simulator-only training layouts and a fixed model-selection suite."""
import math
import random
from ..sim.scenarios import Scenario,BLUE,get_scenario
from ..sim.kinematics import BicycleState


def experiment_case(seed,index,*,held_out=False):
    rng = random.Random(f'held-out:{seed}:{index}' if held_out else seed+index*104729)
    goal = rng.uniform(1.9,2.55),rng.uniform(-.35,.35)
    obstacles = () if index%2==0 else (((rng.uniform(.9,1.3),rng.choice((-1,1))*rng.uniform(.1,.35)),BLUE),)
    return Scenario(f'imitation-{"eval" if held_out else "train"}-{seed}-{index}',obstacles,goal)


def training_case(seed,iteration):
    if iteration < 24:
        return experiment_case(seed,iteration),(0.,0.,0.),'intro'
    families = ('straight','cylinder','turn','slalom','occluded','recovery')
    count = 3 if iteration < 48 else 4 if iteration < 72 else 6
    block = ((iteration-24)//6 if iteration < 48 else
             (iteration-48)//6 if iteration < 72 else (iteration-72)//6)
    family = families[block%count]
    if family in ('straight','cylinder'):
        case = experiment_case(seed,iteration*2+(family=='cylinder'))
        return case,(0.,0.,0.),family
    names = dict(turn='turn_corridor',slalom='slalom',occluded='blocked_goal',recovery='single_cylinder')
    base = get_scenario(names[family])
    rng = random.Random(f'training-curriculum:{seed}:{iteration}')
    mirror = rng.choice((-1,1))
    dx,dy = rng.uniform(-.02,.02),rng.uniform(-.04,.04)
    obstacles = tuple(((x+dx,mirror*y+dy),color) for (x,y),color in base.obstacles)
    goal = base.goal[0]+dx,mirror*base.goal[1]+dy
    pose = (0.,0.,0.)
    if family=='recovery':
        pose = (.65+dx,mirror*(-.30)+dy,mirror*(math.pi/2 if iteration%2 else 0.))
    return Scenario(f'curriculum-{family}-{seed}-{iteration}',obstacles,goal),pose,family


def place_start(env,pose):
    state = BicycleState(*pose)
    if env._pose_collides(state):
        raise ValueError('curriculum start overlaps an obstacle or wall')
    env.start_pose = pose
    env.state = state
    env._sync_kinematic()
    env._last_encoder_position = env._encoder_position()


def validation_cases():
    return [get_scenario(n) for n in ('corridor','single_cylinder','turn_corridor','slalom','blocked_goal')]+[
        experiment_case(77,i,held_out=True) for i in (0,1)]
