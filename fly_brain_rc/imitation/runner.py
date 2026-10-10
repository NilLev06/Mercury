"""Shared imitation collection, held-out evaluation and live telemetry."""
import json
from pathlib import Path
import time
import torch
from ..config import DRIVE,SIM
from ..contracts import ActuatorCommand,BrainTelemetry,RuntimeTelemetry,MotionObservation
from ..perception.target import MockRedBallDetector,MockBlueObstacleDetector
from ..sim.env import FlyCarEnv
from ..sim.scenarios import get_scenario
from .arrival import GoalStop,body_goal_distance
from .connectome import prepare_graph
from .model import RecurrentPolicy,load_checkpoint,save_checkpoint
from .observations import encode_observation
from .teacher import BicycleTeacher
from .training import fit_demonstrations,mix_action,teacher_probability
from .curriculum import experiment_case,training_case,place_start
from .validation import validate,previous_validation,promote_best


def fit_paired_replay(graph,trajectories,*,updates=100,seed=0):
    """Controlled anatomical-vs-dense comparison on identical labelled data.

    Reinitializes both models, not a continuation of either live checkpoint.
    Identical sampler RNG gives identical episode/crop draws to both models.
    """
    models,losses = {},{}
    for name,baseline in (('connectome',False),('rnn',True)):
        torch.manual_seed(seed)
        model = RecurrentPolicy(graph,baseline)
        optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
        torch.manual_seed(seed+1)
        losses[name] = fit_demonstrations(model,optimizer,trajectories,updates=updates)
        models[name] = model
    return models,losses


def collect_episode(env,model,*,steps,beta=None,emit=None,progress=None,
                    cancel=None,viewer=None,mission=None,teacher_factory=None):
    teacher = (teacher_factory or BicycleTeacher)(env) if beta is not None else None
    detector,obstacle_detector = MockRedBallDetector(),MockBlueObstacleDetector()
    state = None
    observations,labels = [],[]
    stop = GoalStop()
    minimum = body_goal_distance(env)
    stalled = 0.0
    action = (0.0,0.0)
    reached,cancelled = False,False
    model.eval()
    for index in range(steps):
        if (cancel and cancel()) or (viewer is not None and not viewer.is_running()):
            cancelled = True
            break
        if index%5==0:
            sensor = env.sensor_frame()
            target = detector.detect(sensor,'red_ball')
            obstacle = obstacle_detector.detect(sensor,'obstacle')
            features = torch.from_numpy(encode_observation(sensor,target,obstacle,
                                        schema=model.observation_schema))
            with torch.no_grad():
                output,state = model(features[None,None],state)
            student = tuple(output[0,0].tolist())
            if teacher is not None:
                action,label = mix_action(teacher.action(),student,beta)
                observations.append(features)
                labels.append(torch.tensor(label,dtype=torch.float32))
            else:
                action = student
            command = ActuatorCommand(*action,sensor.timestamp,index)
            if emit:
                brain = BrainTelemetry(traction=action[0],steering=action[1])
                motion = MotionObservation(0,0,0,0,0,sensor.timestamp)
                emit(RuntimeTelemetry(sensor.timestamp,index,sensor,target,motion,brain,command))
        env.step(*action,observe=False)
        distance = body_goal_distance(env)
        minimum = min(minimum,distance)
        stalled += DRIVE.timestep if abs(env.speed)<.03 and distance>.5 else 0
        reached = stop.update(distance,env.speed,DRIVE.timestep)
        if mission is not None:
            with mission.lock:
                mission.arrival_elapsed = stop.elapsed
                mission.arrival_reason = 'imitation_stop_0.5m' if reached else 'imitation_in_progress'
        if viewer is not None:
            viewer.sync()
            time.sleep(DRIVE.timestep)
        if reached:
            break
    trajectory = (torch.stack(observations),torch.stack(labels)) if observations else None
    return trajectory,{'success':reached,'cancelled':cancelled,'steps':index+1,
                       'contacts':env.collisions,'stalled_seconds':stalled,
                       'min_goal_distance':minimum,'final_goal_distance':body_goal_distance(env),
                       'final_speed':env.speed,'stable_stop_seconds':stop.elapsed,
                       'arrival_criterion':'body-centre-0.5m-speed0.03-hold0.5'}


def run_experiment(*,output,episodes=12,steps=2400,seed=0,evaluate=False,
                   baseline=False,updates=100,data_directory='.local/connectome',
                   emit=None,progress=None,cancel=None,scenario=None,viewer=False,mission=None,
                   validation_interval=12):
    if type(episodes) is not int or episodes<1 or type(steps) is not int or steps<1:
        raise ValueError('positive episode and step counts required')
    if type(validation_interval) is not int or validation_interval<0:
        raise ValueError('nonnegative validation interval required')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    graph = prepare_graph(data_directory)
    path = Path(output)
    if path.exists():
        model,optimizer,data,iteration = load_checkpoint(path,graph,baseline=baseline)
        print(f'Продолжение: {iteration} эпизодов, {len(data)} демонстраций; оптимизатор восстановлен.',flush=True)
        previous_seeds = model.training_metadata['seeds']
        if not evaluate and previous_seeds and seed not in previous_seeds:
            print(f'Новый seed {seed}: добавляем новые карты к прошлым демонстрациям {previous_seeds}.',flush=True)
    elif evaluate:
        raise ValueError('Сначала обучи выбранную imitation-модель.')
    else:
        model = RecurrentPolicy(graph,baseline)
        optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
        data,iteration = [],0
    print(json.dumps({'phase':'loaded','baseline':baseline,'graph':graph['manifest'],
                      'student_inputs':'camera CV + six ranges + ego motion; no hidden goal coordinates'},ensure_ascii=False),flush=True)
    reports = []
    validations = []
    interrupted = False
    def assess():
        print(json.dumps({'phase':'validation_start','iteration':iteration}),flush=True)
        assessment = validate(model,steps=steps,cancel=cancel,emit=emit)
        promoted = promote_best(path,model,optimizer,data,iteration,assessment)
        validations.append(dict(assessment,iteration=iteration,promoted=promoted))
        print(json.dumps({'phase':'validation','validation':validations[-1]},allow_nan=False),flush=True)
        return assessment['cancelled']
    try:
        if not evaluate and validation_interval and previous_validation(path,model,steps) is None:
            interrupted = assess()
        for local_index in range(episodes):
            if interrupted or (cancel and cancel()):
                interrupted = True
                break
            current = local_index if evaluate else iteration
            # Training and test seeds are disjoint, never leak map labels.
            pose,family = (0.,0.,0.),'evaluation' if evaluate else 'explicit'
            if scenario and scenario!='random':
                case = get_scenario(scenario)
            elif evaluate:
                case = experiment_case(seed,current,held_out=True)
            else:
                case,pose,family = training_case(seed,current)
            env = FlyCarEnv(SIM,DRIVE,seed=seed,scenario=case,render=True)
            live_viewer = None
            try:
                place_start(env,pose)
                if viewer:
                    import mujoco.viewer as mj_viewer
                    live_viewer = mj_viewer.launch_passive(env.m,env.d)
                beta = None if evaluate else teacher_probability(iteration)
                if progress:
                    progress(local_index+1,episodes,0,len(data))
                trajectory,report = collect_episode(env,model,steps=steps,beta=beta,emit=emit,
                    cancel=cancel,viewer=live_viewer,mission=mission)
                loss = None
                if not evaluate and trajectory is not None:
                    model.training_metadata['cases'].append(case.name)
                    if seed not in model.training_metadata['seeds']:
                        model.training_metadata['seeds'].append(seed)
                    data.append(trajectory)
                    # Bounded replay buffer; learned weights/optimizer continue.
                    data = data[-128:]
                    loss = fit_demonstrations(model,optimizer,data,updates=updates,cancel=cancel)
                    iteration += 1
                    save_checkpoint(path,model,optimizer,data,iteration)
                report.update(episode=local_index+1,total=episodes,iteration=iteration,
                              evaluation=evaluate,teacher_beta=beta,loss=loss,
                              reward=-(loss or 0),scenario=case.name,baseline=baseline,
                              family=family,start_pose=pose)
                reports.append(report)
                print(json.dumps(report,allow_nan=False),flush=True)
                if progress:
                    progress(local_index+1,episodes,-(loss or 0),len(data))
                if not evaluate:
                    with path.with_suffix('.history.jsonl').open('a') as history:
                        history.write(json.dumps(report,allow_nan=False)+'\n')
                if mission is not None and viewer:
                    mission.finish(report)
                # Keep a completed MuJoCo scene open until the user closes it
                # or presses Stop in the persistent lab. Success already scored.
                if live_viewer is not None and not report['cancelled']:
                    while live_viewer.is_running() and not (cancel and cancel()):
                        live_viewer.sync()
                        time.sleep(.02)
                interrupted = report['cancelled'] or bool(cancel and cancel())
                if interrupted:
                    break
            finally:
                if live_viewer is not None:
                    live_viewer.close()
                if env.renderer is not None:
                    env.renderer.close()
            if not evaluate and validation_interval and iteration%validation_interval==0:
                interrupted = assess()
                if interrupted:
                    break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if not evaluate:
            save_checkpoint(path,model,optimizer,data,iteration)
    result = {'episodes':len(reports),'successes':sum(r['success'] for r in reports),
              'mean_reward':sum(r['reward'] for r in reports)/max(1,len(reports)),
              'cancelled':interrupted,'evaluation':evaluate,'baseline':baseline,
              'reports':reports,'iteration':iteration,'validations':validations}
    if evaluate:
        destination = path.with_name(path.stem+f'.evaluation-{seed}.json')
        destination.write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
