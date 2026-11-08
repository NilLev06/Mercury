import sys, json, math
from pathlib import Path
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from fly_brain_rc.config import SIM, DRIVE
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.sim.scenarios import Scenario
from fly_brain_rc.sim.kinematics import BicycleState
from fly_brain_rc.rl import QLearningAgent, RLController
from fly_brain_rc.bypass_learning import BypassReward
from fly_brain_rc.perception.target import MockRedBallDetector, MockBlueObstacleDetector

root = Path(__file__).parent
agent = QLearningAgent.load(str(ROOT / 'artifacts/rl/bypass-policy.json'))
td, od = MockRedBallDetector(), MockBlueObstacleDetector()
scenario = Scenario('straight-probe', (), (2.3, 0))
camera = FlyCarEnv(SIM, DRIVE, scenario=scenario)
branch = FlyCarEnv(SIM, DRIVE, scenario=scenario, render=False)
rows = []
try:
    for x in (0, .6, 1.2, 1.6, 1.9, 2.1):
        for yaw in (-.65, -.4, -.2, -.12, -.06, 0, .06, .12, .2, .4, .65):
            initial = BicycleState(x, 0, yaw, .105, 0)
            camera.state = initial
            camera.speed = initial.speed
            camera._sync_kinematic()
            camera._last_encoder_position = camera._encoder_position()
            sensor = camera.sensor_frame()
            target = td.detect(sensor, 'red_ball')
            if target is None:
                continue
            state = RLController(agent).observe(sensor, target, od.detect(sensor, 'obstacle'))
            scores = []
            for traction, steering in agent.actions:
                branch.reset()
                branch.state = initial
                branch.speed = initial.speed
                branch._sync_kinematic()
                reward = BypassReward(branch.goal_distance())
                score = 0
                for tick in range(50):
                    branch.step(traction, steering, observe=False)
                    reached = branch.goal_center_reached()
                    score += reward.step(branch.goal_distance(), contact=branch._contact_active,
                                         steering=steering, reached=reached, dt=.02)
                    if reached:
                        break
                scores.append(score)
            rows.append(dict(x=x, yaw=yaw, bearing=target.bearing, area=target.apparent_area,
                             state=state, learned_action=agent.choose(state, explore=False),
                             best_one_second_action=max(range(len(scores)), key=scores.__getitem__),
                             scores=scores))
finally:
    camera.renderer.close()
groups = {}
for row in rows:
    groups.setdefault(tuple(row['state']), []).append(row)
conflicts = [g for g in groups.values() if len({r['best_one_second_action'] for r in g}) > 1]
(root/'alias-probes.json').write_text(json.dumps(dict(rows=rows, conflicting_groups=conflicts), indent=2))
print('PROBES', len(rows), 'STATES', len(groups), 'CONFLICT_GROUPS', len(conflicts))
for group in conflicts[:4]:
    print('SAME_STATE', group[0]['state'])
    for r in group:
        print('DETAIL', {k: r[k] for k in ('x','yaw','bearing','area','learned_action','best_one_second_action')})
