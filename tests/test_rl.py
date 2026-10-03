import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

from fly_brain_rc.rl import ExplorationReward, QLearningAgent, QLearningBrain, observation_state
from fly_brain_rc.rl_training import observation_state
from fly_brain_rc.rl_training import load_training_agent
from fly_brain_rc.rl_training import train_q_learning
from tests.test_environment import HeadlessRenderer
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.contracts import ObjectObservation, SensorFrame
import numpy as np


class ReinforcementLearningTests(unittest.TestCase):
    def test_memory_confidence_decays_in_telemetry(self):
        b = QLearningBrain(QLearningAgent())
        s = SensorFrame(np.zeros((2,2,3),dtype=np.uint8),np.zeros((2,2)),0,0,1.2,False,0)
        goal = ObjectObservation('red_ball',1,(.4,.4,.6,.6),0,.04,0)
        _, visible = b.step(s,goal,None,0,.02)
        _, recalled = b.step(replace(s,timestamp=6),None,None,0,.02)
        _, expired = b.step(replace(s,timestamp=13),None,None,0,.02)
        self.assertEqual(visible.populations['rl_memory_confidence'],1)
        self.assertAlmostEqual(recalled.populations['rl_memory_confidence'],np.exp(-1))
        self.assertEqual(expired.populations['rl_memory_confidence'],0)

    def test_recalled_state_warm_start_does_not_mutate_visible_values(self):
        a = QLearningAgent()
        visible = ('target_left','clear','small','clear','clear','clear','forward','no_contact','no_contact')
        recalled = ('target_memory_left',)+visible[1:]
        a.table[visible] = [2.0]*len(a.actions)
        self.assertEqual(a.values(recalled),a.values(visible))
        self.assertNotIn(recalled,a.table)
        a.value(recalled,0)
        a.table[recalled][0] = 3.0
        self.assertEqual(a.table[visible][0],2.0)

    def test_contact_recovery_overrides_bypass_and_resumes_along(self):
        from fly_brain_rc.rl import RLController
        c = RLController(QLearningAgent())
        s = SensorFrame(np.zeros((2,2,3),dtype=np.uint8),np.zeros((2,2)),0,0,.6,False,0)
        obstacle = ObjectObservation('obstacle',1,(.3,.3,.7,.8),0,.1,0)
        state = c.observe(s,None,obstacle)
        action = c.choose(state,explore=False)
        self.assertEqual(c.mode,'bypass_outward')
        self.assertGreater(c.agent.actions[action][0],0)
        state = c.observe(replace(s,timestamp=.02,contact=True,contact_front=True),None,obstacle)
        action = c.choose(state,explore=False)
        self.assertEqual(c.mode,'retreat')
        self.assertLess(c.agent.actions[action][0],0)
        self.assertTrue(c.overriding)
        # Physical encoder progress completes the retreat, arc and clear.
        c.observe(replace(s,timestamp=.04,encoder_delta=-.4),None,obstacle)
        c.observe(replace(s,timestamp=1.04,encoder_delta=.4,yaw_rate=c.recovery.turn*1.2),None,obstacle)
        c.observe(replace(s,timestamp=1.06,encoder_delta=.4),None,None)
        self.assertEqual(c.mode,'bypass_along')

    def test_recalled_direction_has_directional_fallback(self):
        a = QLearningAgent()
        state = ('target_memory_left','clear','small','clear','clear','clear','forward','no_contact','no_contact')
        action = a.choose(state,explore=False)
        self.assertLess(a.actions[action][1],0)

    def test_brain_reset_clears_target_and_bypass_memory(self):
        b = QLearningBrain(QLearningAgent())
        s = SensorFrame(np.zeros((2,2,3),dtype=np.uint8),np.zeros((2,2)),0,0,.6,False,0)
        goal = ObjectObservation('red_ball',1,(.4,.4,.6,.6),0,.04,0)
        obstacle = ObjectObservation('obstacle',1,(.3,.3,.7,.8),0,.1,0)
        b.step(s,goal,None,0,.02,obstacle=obstacle)
        b.reset()
        self.assertIsNone(b.controller.navigation.vector)
        self.assertEqual(b.controller.navigation.phase,'drive')

    def test_target_memory_turns_with_gyro_and_expires(self):
        from fly_brain_rc.rl import RLController
        c = RLController(QLearningAgent())
        s = SensorFrame(np.zeros((2,2,3),dtype=np.uint8),np.zeros((2,2)),0,0,1.2,False,0)
        goal = ObjectObservation('red_ball',1,(.4,.4,.6,.6),0,.04,0)
        c.observe(s,goal,None)
        state = c.observe(replace(s,timestamp=1,yaw_rate=.4),None,None)
        self.assertEqual(state[0],'target_memory_left')
        state = c.observe(replace(s,timestamp=13),None,None)
        self.assertEqual(state[0],'target_none')

    def test_target_memory_updates_range_with_encoder_without_world_pose(self):
        from fly_brain_rc.rl import RLController
        c = RLController(QLearningAgent())
        s = SensorFrame(np.zeros((2,2,3),dtype=np.uint8),np.zeros((2,2)),0,0,1.2,False,0)
        goal = ObjectObservation('red_ball',1,(.4,.4,.6,.6),.5,.04,0)
        c.observe(s,goal,None)
        before = c.navigation.goal_bearing
        c.observe(replace(s,timestamp=.2,encoder_delta=.2),None,None)
        self.assertGreater(c.navigation.goal_bearing,before)

    def test_bypass_side_is_retained_when_ranges_change(self):
        from fly_brain_rc.rl import RLController
        c = RLController(QLearningAgent())
        s = SensorFrame(np.zeros((2,2,3),dtype=np.uint8),np.zeros((2,2)),0,0,.6,False,0,
                        range_left=1.2,range_right=.4)
        obstacle = ObjectObservation('obstacle',1,(.3,.3,.7,.8),0,.1,0)
        c.observe(s,None,obstacle)
        side = c.navigation.side
        c.observe(replace(s,timestamp=.2,range_left=.2,range_right=1.2),None,obstacle)
        self.assertEqual(c.navigation.side,side)
        self.assertNotEqual(side,0)

    def test_v6_migration_retains_policy_and_training_count(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'policy.json'
            key = 'target_center|clear|small|clear|clear|clear|forward|no_contact|no_contact'
            old = json.dumps(dict(version=6,actions=[[.3,0],[-.7,0]],alpha=.18,gamma=.98,
                                 training_runs=2,table={key:[4,-2]}))
            p.write_text(old)
            a = load_training_agent(p,seed=0,epsilon=.25)
            self.assertEqual(a.table[tuple(key.split('|'))],[4,-2])
            self.assertEqual(a.training_runs,2)
            self.assertEqual(p.with_name('policy.json.v6.bak').read_text(),old)

    def test_centered_visible_goal_does_not_start_range_avoidance(self):
        from fly_brain_rc.rl import RLController
        c = RLController(QLearningAgent())
        s = SensorFrame(np.zeros((2,2,3),dtype=np.uint8),np.zeros((2,2)),0,0,.15,False,0)
        goal = ObjectObservation('red_ball',1,(.3,.2,.95,.8),.31,.4,0)
        c.action = 5
        c.observe(s,goal,None)
        self.assertEqual(c.recovery.phase,'drive')

    def test_contact_interrupts_held_forward_action_immediately(self):
        s = SensorFrame(np.zeros((2,2,3),dtype=np.uint8),np.zeros((2,2)),0,0,1.2,False,0)
        brain = QLearningBrain(QLearningAgent())
        self.assertGreater(brain.step(s,None,None,0,.02)[0].traction,0)
        command = brain.step(replace(s,timestamp=.02,contact=True,contact_front=True),None,None,0,.02)[0]
        self.assertLess(command.traction,0)

    def test_recovery_commands_are_not_learned_as_normal_actions(self):
        with tempfile.TemporaryDirectory() as directory, patch('fly_brain_rc.sim.env.mujoco.Renderer', HeadlessRenderer):
            p = Path(directory)/'policy.json'
            train_q_learning(episodes=1,steps=100,output=p,epsilon=0)
            a = QLearningAgent.load(p)
            # One pursuit window initiates avoidance. The following recovery
            # windows must not populate Q with scripted reverse/arc commands.
            self.assertEqual(len(a.table),1)
            self.assertEqual(sum(v != 0 for row in a.table.values() for v in row),1)

    def test_training_bypass_does_not_update_q_or_reward_recalled_target(self):
        goal = ObjectObservation('red_ball',1,(.4,.4,.6,.6),0,.04,0)
        obstacle = ObjectObservation('obstacle',1,(.3,.3,.7,.8),0,.1,0)
        frames = []
        def sensor_frame(env):
            return SensorFrame(np.zeros((2,2,3),dtype=np.uint8),np.zeros((2,2)),
                               0,0,.6,False,env.sim_steps*env.drive.timestep)
        with tempfile.TemporaryDirectory() as directory, \
             patch('fly_brain_rc.sim.env.mujoco.Renderer', HeadlessRenderer), \
             patch.object(FlyCarEnv,'sensor_frame',sensor_frame), \
             patch('fly_brain_rc.rl_training.MockRedBallDetector.detect',side_effect=[goal]+[None]*20), \
             patch('fly_brain_rc.rl_training.MockBlueObstacleDetector.detect',return_value=obstacle), \
             patch.object(ExplorationReward,'step',autospec=True,return_value=0) as reward:
            p = Path(directory)/'policy.json'
            train_q_learning(episodes=1,steps=20,output=p,epsilon=0,emit=frames.append)
            self.assertEqual(QLearningAgent.load(p).table,{})
            self.assertTrue(all(f.brain.populations['rl_target_memory'] == 1 for f in frames))
            self.assertTrue(all(call.kwargs['target'] is None for call in reward.call_args_list))

    def test_contact_release_leads_to_committed_arc_not_pursuit(self):
        from fly_brain_rc.rl import RLController
        a = QLearningAgent()
        c = RLController(a)
        s = SensorFrame(np.zeros((2,2,3), dtype=np.uint8), np.zeros((2,2)),0,0,1.2,True,0,contact_front=True)
        state = c.observe(s,None,None)
        self.assertLess(a.actions[c.choose(state,explore=False)][0],0)
        state = c.observe(replace(s,timestamp=3.8,contact=False,contact_front=False,encoder_delta=-.9),None,None)
        command = a.actions[c.choose(state,explore=False)]
        self.assertGreater(command[0],0)
        self.assertGreater(abs(command[1]),.5)
        self.assertTrue(c.overriding)

    def test_compact_key_does_not_split_by_previous_command(self):
        s = SensorFrame(np.zeros((2,2,3), dtype=np.uint8), np.zeros((2,2)),0,0,1.2,False,0)
        self.assertEqual(observation_state(s,None,None,previous='forward_left',failed='forward_failed',blocked_action=3),
                         observation_state(s,None,None,previous='reverse_right'))

    def test_v5_migration_merges_estimates_without_erasing_training(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'policy.json'
            base = 'target_none|clear|small|clear|clear|clear|forward|no_contact|no_contact'
            old = json.dumps(dict(version=5,actions=[[.3,0],[-.7,0]],alpha=.18,gamma=.98,training_runs=1,
                                 table={base+'|forward_left|none|none':[2,0],base+'|forward_right|none|none':[4,-2],
                                        base+'|reverse_right|forward_failed|5':[100,100]}))
            p.write_text(old)
            a = load_training_agent(p,seed=0,epsilon=.25)
            self.assertEqual(len(a.table),1)
            self.assertEqual(a.table[tuple(base.split('|'))],[3,-2])
            self.assertEqual(a.training_runs,1)
            self.assertEqual(p.with_name('policy.json.v5.bak').read_text(),old)

    def test_migration_retains_estimates_when_ordinary_values_are_zero(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'policy.json'
            base = 'target_none|clear|small|clear|clear|clear|forward|no_contact|no_contact'
            p.write_text(json.dumps(dict(version=5,actions=[[.3,0],[-.7,0]],alpha=.18,gamma=.98,
                table={base+'|forward_left|none|none':[0,0],base+'|reverse_right|forward_failed|5':[10,-2]})))
            self.assertEqual(QLearningAgent.load(p).table[tuple(base.split('|'))],[10,-2])

    def test_front_wall_escape_changes_heading_and_gets_clear(self):
        from fly_brain_rc.rl import RLController
        from fly_brain_rc.sim.kinematics import BicycleState
        from fly_brain_rc.sim.scenarios import Scenario
        from fly_brain_rc.config import SIM,DRIVE
        from fly_brain_rc.contracts import ActuatorCommand
        with patch('fly_brain_rc.sim.env.mujoco.Renderer', HeadlessRenderer):
            env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('wall',(),(0,.6)))
            env.state = BicycleState(2.77,0,0)
            env._sync_kinematic()
            c = RLController(QLearningAgent())
            for i in range(800):
                s = env.sensor_frame()
                state = c.observe(s,None,None)
                if i%10==0: action = c.choose(state,explore=False)
                t,w = c.agent.actions[action]
                env.apply(ActuatorCommand(t,w,s.timestamp,i))
            self.assertLess(env.car_pose()[0],2.4)
            self.assertGreater(abs(env.car_pose()[2]),.8)
            env.renderer.close()

    def test_recovery_memory_survives_brief_contact_release(self):
        from fly_brain_rc.rl import RLController
        agent = QLearningAgent()
        controller = RLController(agent)
        sensor = SensorFrame(np.zeros((2,2,3), dtype=np.uint8), np.zeros((2,2)), 0,0,1.2,False,0)
        controller.observe(sensor, None, None)
        controller.action = 5
        controller.observe(replace(sensor, timestamp=.02, contact=True, contact_front=True), None, None)
        state = controller.observe(replace(sensor, timestamp=.22, encoder_delta=.015), None, None)
        self.assertEqual(controller.recovery.phase, 'retreat')
        self.assertLess(agent.actions[controller.choose(state, explore=False)][0], 0)

    def test_bootstrap_ignores_direction_blocked_by_bumper(self):
        agent = QLearningAgent(alpha=1, gamma=.5, actions=((.3,0),(-.7,0)))
        next_state = ('target_none','clear','small','clear','clear','clear','stalled','contact_center',
                      'contact_front','none','none','none')
        agent.table[next_state] = [100,-10]
        agent.update(('prior',), 0, 0, next_state, terminal=False)
        self.assertEqual(agent.value(('prior',),0), -5)

    def test_contact_classification_uses_colliding_not_nearest_obstacle(self):
        from fly_brain_rc.sim.scenarios import Scenario, BLUE
        from fly_brain_rc.sim.kinematics import BicycleState
        from fly_brain_rc.config import SIM, DRIVE
        env = FlyCarEnv(SIM, DRIVE, scenario=Scenario('two', (((.5005,0),BLUE), ((-.322,0),BLUE)), (2,0)), render=False)
        hit, _ = env._motion_collides(BicycleState(0,0,0), BicycleState(.001,0,0))
        self.assertTrue(hit)
        self.assertGreater(env._collision_longitudinal, 0)

    def test_unknown_frozen_choices_are_deterministic_and_forward(self):
        sensor = SensorFrame(np.zeros((2,2,3), dtype=np.uint8), np.zeros((2,2)), 0, 0, 1.2, False, 0)
        commands = [QLearningBrain(QLearningAgent(seed=i)).step(sensor, None, None, 0, .02)[0]
                    for i in range(20)]
        self.assertEqual(len({(c.traction, c.steering) for c in commands}), 1)
        self.assertGreater(commands[0].traction, 0)

    def test_repeated_blocked_reverse_is_replaced_by_forward(self):
        sensor = SensorFrame(np.zeros((2,2,3), dtype=np.uint8), np.zeros((2,2)),
                             0, 0, 1.2, True, 0, contact_right=True)
        learner = QLearningAgent()
        learner.table[observation_state(sensor, None, None)] = [0]*8 + [10, 0, 0]
        brain = QLearningBrain(learner)
        first = brain.step(sensor, None, None, 0, .02)[0]
        self.assertLess(first.traction, 0)
        for tick in range(1, 21):
            command = brain.step(replace(sensor, timestamp=tick*.02), None, None, 0, .02)[0]
        self.assertGreater(command.traction, 0)

    def test_rear_contact_does_not_repeat_failed_forward_turn(self):
        sensor = SensorFrame(np.zeros((2,2,3), dtype=np.uint8), np.zeros((2,2)),
                             0, 0, 1.2, True, 0, contact_right=True, contact_rear=True)
        agent = QLearningAgent()
        from fly_brain_rc.rl import RLController
        controller = RLController(agent)
        controller.action = 2
        state = controller.observe(sensor,None,None)
        agent.table[state] = [0, 0, 99, 0, 0, 0, 0, 100, 0, 0, 0]
        action = controller.choose(state, explore=False)
        self.assertGreater(agent.actions[action][0], 0)
        self.assertLessEqual(agent.actions[action][1], .1)

    def test_front_and_rear_contacts_have_different_policy_states(self):
        from fly_brain_rc.sim.scenarios import Scenario, BLUE
        from fly_brain_rc.sim.kinematics import BicycleState
        from fly_brain_rc.config import SIM, DRIVE
        states = []
        for x, throttle in [(0, 1), (.84, -1)]:
            env = FlyCarEnv(SIM, DRIVE, scenario=Scenario('bumper', (((.5,0), BLUE),), (2,0)), render=False)
            env.state = BicycleState(x,0,0,0)
            for _ in range(30): env.step_kinematic(throttle, 0, .02)
            states.append((getattr(env, '_contact_front', False), getattr(env, '_contact_rear', False)))
        self.assertEqual(states, [(True, False), (False, True)])

    def test_training_reads_once_per_tick_and_learns_forward_states(self):
        instances = []
        def make_env(*args, **kwargs):
            env = FlyCarEnv(*args, **kwargs)
            env.sensor_frame = unittest.mock.Mock(wraps=env.sensor_frame)
            instances.append(env)
            return env
        with tempfile.TemporaryDirectory() as directory, \
             patch('fly_brain_rc.sim.env.mujoco.Renderer', HeadlessRenderer), \
             patch('fly_brain_rc.rl_training.FlyCarEnv', side_effect=make_env), \
             patch.object(QLearningAgent, 'choose', return_value=0):
            path = Path(directory) / 'policy.json'
            train_q_learning(episodes=1, steps=20, output=path)
            learner = QLearningAgent.load(path)
            self.assertEqual(instances[0].sensor_frame.call_count, 21)
            self.assertTrue(learner.table)
            import json
            reports = [json.loads(line) for line in path.with_suffix('.history.jsonl').read_text().splitlines()]
            self.assertEqual(len(reports), 1)
            self.assertEqual(reports[0]['steps'], 20)
            self.assertIn('contact_seconds', reports[0])
            self.assertIn('reverse_distance', reports[0])

    def test_policy_holds_action_despite_changed_observation_until_point_two_seconds(self):
        learner = QLearningAgent(actions=((.3, -.6), (.3, .6)), epsilon=0)
        sensor = SensorFrame(np.zeros((2,2,3), dtype=np.uint8), np.zeros((2,2), dtype=np.uint8),
                             0, 0, .8, False, 0)
        other = replace(sensor, tof=.5, timestamp=.02)
        learner.table[observation_state(sensor, None, None)] = [2, 0]
        learner.table[observation_state(other, None, None)] = [0, 2]
        brain = QLearningBrain(learner)
        self.assertLess(brain.step(sensor, None, None, 0, .02)[0].steering, 0)
        self.assertLess(brain.step(other, None, None, 0, .02)[0].steering, 0)
        self.assertGreater(brain.step(replace(other, timestamp=.2), None, None, 0, .02)[0].steering, 0)

    def test_old_table_is_preserved_when_observation_schema_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            original = '{"version": 2, "table": {"old": [42]}}'
            path.write_text(original)
            agent = load_training_agent(path, seed=0, epsilon=.25)
            self.assertEqual(agent.table, {})
            self.assertEqual(path.with_name('policy.json.v2.bak').read_text(), original)

    def test_new_cell_is_rewarded_and_stationary_streak_is_penalized(self):
        reward = ExplorationReward(cell_size=0.5)
        reward.reset((0.0, 0.0))

        discovered = reward.step((0.6, 0.0), moved=True, contact=False, reached=False)
        first_idle = reward.step((0.6, 0.0), moved=False, contact=False, reached=False)
        repeated_idle = reward.step((0.6, 0.0), moved=False, contact=False, reached=False)

        self.assertGreater(discovered, 0.0)
        self.assertLess(first_idle, 0.0)
        self.assertLess(repeated_idle, first_idle)

    def test_only_target_improvement_is_rewarded(self):
        reward = ExplorationReward()
        reward.reset((0.0, 0.0))
        centered = ObjectObservation("red_ball", 1.0, (.4, .4, .6, .6), 0.0, .05, 0.0)

        no_target = reward.step((0.0, 0.0), moved=True, contact=False, reached=False, target=None)
        target_reward = reward.step((0.0, 0.0), moved=True, contact=False, reached=False, target=centered)

        self.assertEqual(target_reward, no_target)
        unchanged = reward.step((0, 0), moved=True, contact=False, reached=False, target=centered)
        closer = reward.step((0, 0), moved=True, contact=False, reached=False,
                             target=replace(centered, apparent_area=.06))
        self.assertEqual(unchanged, 0)
        self.assertGreater(closer, 0)

    def test_contact_release_gives_no_repeatable_bonus(self):
        reward = ExplorationReward()
        reward.step((0, 0), moved=True, contact=True, reached=False)
        self.assertEqual(reward.step((0, 0), moved=True, contact=False, reached=False), 0)

    def test_short_reverse_is_allowed_but_prolonged_reverse_costs_more(self):
        reward = ExplorationReward()
        values = [reward.step((0, 0), moved=True, contact=False, reached=False, speed=-.2, dt=.02)
                  for _ in range(200)]
        self.assertEqual(sum(values[:25]), 0)
        self.assertLess(sum(values[50:100]), -.2)
        self.assertLess(sum(values[150:200]), sum(values[50:100]))

    def test_stall_penalty_is_capped_and_contact_is_only_charged_on_impact(self):
        reward = ExplorationReward(idle_penalty=.01, max_idle_penalty=.12, contact_penalty=1.5)
        reward.reset((0.0, 0.0))

        tail = [reward.step((0.0, 0.0), moved=False, contact=False, reached=False) for _ in range(200)]
        impact = reward.step((0.0, 0.0), moved=True, contact=True, reached=False)
        held_contact = reward.step((0.0, 0.0), moved=True, contact=True, reached=False)

        self.assertGreaterEqual(min(tail), -.12)
        self.assertAlmostEqual(impact, -1.5)
        self.assertEqual(held_contact, 0.0)

    def test_positive_experience_increases_selected_action_value(self):
        learner = QLearningAgent(actions=((0.6, 0.0),), alpha=0.5, gamma=0.0, epsilon=0.0)
        state = ("target_center", "clear", "moving")

        learner.update(state, 0, 2.0, state, terminal=True)

        self.assertGreater(learner.value(state, 0), 0.0)

    def test_training_reuses_saved_table_instead_of_starting_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            state = ("target_none", "clear", "far", "moving", "no_contact")
            learner = QLearningAgent(epsilon=0.0)
            learner.update(state, 0, 2.0, state, terminal=True)
            learner.save(path)

            resumed = load_training_agent(path, seed=7, epsilon=.25)

            self.assertGreater(resumed.value(state, 0), 0.0)
            self.assertEqual(resumed.epsilon, .25)
            self.assertEqual(resumed.training_runs, 0)

    def test_policy_state_uses_perception_but_not_world_coordinates(self):
        sensor = SensorFrame(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8),
                             0.0, 0.002, 0.8, False, 0.0)
        target = ObjectObservation("red_ball", 1.0, (72, 20, 20, 20), 0.4, 0.08, 0.0)

        state = observation_state(sensor, target, obstacle=None)

        self.assertEqual(state, ("target_right", "clear", "small", "clear", "clear", "clear", "stalled", "no_contact", "no_contact"))

    def test_policy_state_distinguishes_the_side_that_needs_an_escape_turn(self):
        sensor = SensorFrame(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8),
                             0.0, 0.0, 0.2, True, 0.0, contact_left=True)

        state = observation_state(sensor, target=None, obstacle=None)

        self.assertEqual(state[7], "contact_left")

    def test_recovery_curriculum_includes_walls_and_both_contact_directions(self):
        from fly_brain_rc.sim import suites
        cases = suites.recovery_cases(0, 4)
        self.assertEqual(len(cases), 4)
        self.assertEqual([c.scenario.name.split('-')[0] for c in cases],
                         ['recover_front_cylinder', 'recover_rear_cylinder', 'recover_front_wall', 'recover_rear_wall'])
        self.assertGreater(cases[2].start_pose[0], 2.7)
        self.assertLess(cases[3].start_pose[0], -2.9)

    def test_v4_training_starts_new_table_and_preserves_bytes(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            old = json.dumps(dict(version=4, table={'old': [9]}))
            path.write_text(old)
            learner = load_training_agent(path, seed=0, epsilon=.25)
            self.assertEqual(learner.table, {})
            self.assertEqual(path.with_name('policy.json.v4.bak').read_text(), old)

    def test_frozen_policy_turns_the_action_with_highest_learned_value_into_motor_command(self):
        agent = QLearningAgent(actions=((0.7, -0.4), (0.7, 0.4)), epsilon=0.0)
        sensor = SensorFrame(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8),
                             0.0, 0.0, 0.8, False, 0.0)
        state = observation_state(sensor, None, None)
        agent.table[state] = [0.1, 2.0]

        command, telemetry = QLearningBrain(agent).step(sensor, None, None, 0.0, 0.02, obstacle=None)

        self.assertEqual((command.traction, command.steering), (0.7, 0.4))
        self.assertEqual(telemetry.populations["rl_best_value"], 2.0)


if __name__ == "__main__":
    unittest.main()
