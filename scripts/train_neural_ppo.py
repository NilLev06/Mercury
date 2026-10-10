"""Isolated on-policy PPO motor-adapter experiment; never promotes live weights."""
import argparse,hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from fly_brain_rc.config import SIM,DRIVE
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.imitation.arrival import GoalStop,body_goal_distance
from fly_brain_rc.imitation.curriculum import place_start,validation_cases
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from fly_brain_rc.imitation.model import load_checkpoint,save_checkpoint
from fly_brain_rc.imitation.observations import encode_observation
from fly_brain_rc.perception.target import MockRedBallDetector,MockBlueObstacleDetector
from scripts.train_sequential import episode,evaluate
from scripts.improve_navigation import write


def freeze_actor(model):
    if model.motor_head!='residual32':raise ValueError('residual32 actor required')
    model.trainable_core=False
    for name,p in model.named_parameters():p.requires_grad_('readout.residual.' in name)


def squashed_log_prob(z,mean,std):
    log_jacobian=2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))
    return (torch.distributions.Normal(mean,std).log_prob(z)-log_jacobian).sum(-1)


def clipped_loss(new_logp,old_logp,adv,clip=.2):
    ratio=(new_logp-old_logp).exp()
    return -torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv)


def advantages(reward,value,*,bootstrap,terminal,gamma=.995,lam=.95):
    result=torch.zeros_like(reward);last=reward.new_zeros(())
    following=0. if terminal else bootstrap
    for i in range(len(reward)-1,-1,-1):
        delta=reward[i]+gamma*following-value[i]
        last=delta+gamma*lam*last;result[i]=last;following=value[i]
    return result,result+value


def transition_reward(d0,d1,*,reached,contact,contact_seconds,elapsed,gamma=.995):
    terminal=reached or contact
    parts=dict(potential=5*(d0-(0. if terminal else gamma*d1)),
               arrival=25.*bool(reached),collision=-10.*bool(contact),
               contact_duration=-.2*contact_seconds,time=-.02*elapsed)
    return sum(parts.values()),parts


class Critic(nn.Module):
    def __init__(self,n):
        super().__init__();self.net=nn.Sequential(nn.Linear(n,64),nn.Tanh(),nn.Linear(64,1))
    def forward(self,h):return self.net(h).squeeze(-1)


def brain_step(model,env,state,detectors):
    sensor=env.sensor_frame()
    target=detectors[0].detect(sensor,'red_ball');obstacle=detectors[1].detect(sensor,'obstacle')
    x=torch.from_numpy(encode_observation(sensor,target,obstacle,schema=model.observation_schema))
    _,state=model(x[None,None],state)
    return state


def rollout(model,critic,case,pose,std,*,steps=2400,gamma=.995):
    """No teacher. Terminal first contact; time limit is a bootstrapped cutoff."""
    env=None;hs=[];zs=[];ps=[];vs=[];rewards=[];components={}
    try:
        env=FlyCarEnv(SIM,DRIVE,scenario=case,render=True);place_start(env,pose)
        detectors=(MockRedBallDetector(),MockBlueObstacleDetector());state=None;stop=GoalStop()
        reached=contact=False;used=0;minimum=body_goal_distance(env)
        with torch.no_grad():
            while used<steps:
                state=brain_step(model,env,state,detectors);h=state[0]
                mean=model.readout(h);z=mean+torch.randn_like(mean)*std;action=z.tanh()
                hs.append(h.clone());zs.append(z);ps.append(squashed_log_prob(z,mean,std));vs.append(critic(h))
                before=body_goal_distance(env);contact_seconds=0.;elapsed=0.
                for _ in range(min(5,steps-used)):
                    env.step(*action.tolist(),observe=False);used+=1;elapsed+=DRIVE.timestep
                    contact=bool(env._contact_active)
                    contact_seconds+=DRIVE.timestep*contact
                    distance=body_goal_distance(env);minimum=min(minimum,distance)
                    reached=stop.update(distance,env.speed,DRIVE.timestep)
                    if reached or contact:break
                reward,parts=transition_reward(before,distance,reached=reached,contact=contact,
                    contact_seconds=contact_seconds,elapsed=elapsed,gamma=gamma)
                rewards.append(reward)
                for k,v in parts.items():components[k]=components.get(k,0.)+v
                if reached or contact:break
            terminal=reached or contact
            bootstrap=0. if terminal else float(critic(brain_step(model,env,state,detectors)[0]))
        value=torch.stack(vs);adv,returns=advantages(torch.tensor(rewards),value,bootstrap=bootstrap,terminal=terminal,gamma=gamma)
        report=dict(scenario=case.name,success=reached,contacts=env.collisions,steps=used,
            final_goal_distance=body_goal_distance(env),min_goal_distance=minimum,final_speed=env.speed,
            stable_stop_seconds=stop.elapsed,terminal=terminal,truncated=not terminal,
            reward=sum(rewards),reward_parts=components)
        return dict(h=torch.stack(hs),z=torch.stack(zs),old_logp=torch.stack(ps),value=value,adv=adv,returns=returns),report
    finally:
        if env is not None and env.renderer is not None:env.renderer.close()


def update(model,critic,actor_opt,critic_opt,episodes,std,anchors,*,epochs=4,anchor_weight=1.):
    keys=('h','z','old_logp','value','adv','returns');batch={k:torch.cat([e[k] for e in episodes]) for k in keys}
    n=len(batch['h']);weights=torch.cat([torch.full((len(e['h']),),1/(len(episodes)*len(e['h']))) for e in episodes])
    adv=batch['adv'];center=(weights*adv).sum();scale=(weights*(adv-center).square()).sum().sqrt().clamp_min(1e-6)
    batch['adv']=(adv-center)/scale;stats=[];stopped=False
    for epoch in range(epochs):
        for index in torch.randperm(n).split(256):
            h=batch['h'][index];logp=squashed_log_prob(batch['z'][index],model.readout(h),std)
            w=weights[index]*n
            loss=(w*clipped_loss(logp,batch['old_logp'][index],batch['adv'][index])).mean()
            anchor_loss=loss.new_zeros(())
            if anchors is not None:
                ah,ay=anchors;a=torch.randint(len(ah),(256,))
                anchor_loss=(model.readout(ah[a]).tanh()-ay[a]).square().mean()
                loss=loss+anchor_weight*anchor_loss
            if not torch.isfinite(loss):raise ValueError('nonfinite PPO loss')
            actor_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.readout.residual.parameters(),1.);actor_opt.step()
            value_loss=(w*(critic(h)-batch['returns'][index]).square()).mean()
            critic_opt.zero_grad();value_loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),10.);critic_opt.step()
            with torch.no_grad():
                new=squashed_log_prob(batch['z'],model.readout(batch['h']),std)
                log_ratio=new-batch['old_logp']
                kl=float((weights*(log_ratio.exp()-1-log_ratio)).sum())
            stats.append(dict(loss=float(loss.detach()),value_loss=float(value_loss.detach()),kl=kl,anchor=float(anchor_loss.detach())))
            if kl>.03:stopped=True;break
        if stopped:break
    return dict(updates=len(stats),early_stop=stopped,last=stats[-1])


def make_anchors(model,cases):
    hs=[];ys=[];reports=[]
    for case in cases:
        captured=[]
        hook=model.readout.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
        try:_,_,report=episode(model,case,(0.,0.,0.))
        finally:hook.remove()
        reports.append(report)
        if report['success'] and report['contacts']==0:
            h=torch.cat(captured)
            with torch.no_grad():y=model.readout(h).tanh()
            hs.append(h);ys.append(y)
    if not hs:raise ValueError('source has no successful old anchors')
    return (torch.cat(hs),torch.cat(ys)),reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('source','out'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=40);parser.add_argument('--evaluate-every',type=int,default=5)
    parser.add_argument('--lr',type=float,default=3e-5);parser.add_argument('--anchor-weight',type=float,default=1.)
    parser.add_argument('--seed',type=int,default=10092026)
    args=parser.parse_args()
    if args.rounds<1 or args.evaluate_every<1 or args.lr<=0 or args.anchor_weight<0:raise ValueError('invalid experiment settings')
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'run.py').write_text(Path(__file__).read_text())
    source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest();payload=torch.load(args.source,weights_only=True)
    torch.set_num_threads(2);model,_,_,age=load_checkpoint(args.source,payload['graph']);freeze_actor(model)
    torch.manual_seed(args.seed);model.eval();frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    critic=Critic(len(model.graph['ids']));actor_opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    critic_opt=torch.optim.Adam(critic.parameters(),lr=1e-3)
    old_cases=validation_cases();dev=mixed_cases(20001,'selection',1)
    anchors,old_reports=make_anchors(model,old_cases);torch.save(dict(h=anchors[0],y=anchors[1]),args.out/'anchors.pt')
    write(args.out/'source-development.json',dict(old=old_reports,new=evaluate(model,dev)))
    results=[];collection=[]
    for round_index in range(args.rounds):
        torch.manual_seed(args.seed+1000+round_index)
        std=torch.tensor([.06,.3-.15*min(1,round_index/30)])
        cases=[(old_cases[(round_index*4+i)%7],(0.,0.,0.),'old') for i in range(4)]
        cases+=mixed_cases(60000+round_index,'train',2)
        episodes=[]
        for case,pose,tier in cases:
            data,report=rollout(model,critic,case,pose,std);episodes.append(data)
            report.update(round=round_index,tier=tier);collection.append(report)
            print(json.dumps(dict(phase='ppo_episode',**report)),flush=True)
        # Raw on-policy rollout archive documents the exact behavior policy.
        torch.save(dict(episodes=episodes,std=std,torch_rng=torch.get_rng_state()),args.out/'last-rollout.pt')
        stats=update(model,critic,actor_opt,critic_opt,episodes,std,anchors,anchor_weight=args.anchor_weight)
        assert all(torch.equal(dict(model.named_parameters())[n],p) for n,p in frozen.items())
        model.training_metadata.pop('fit_scope',None)
        model.training_metadata['neural_ppo']=dict(source=str(args.source.resolve()),source_sha256=source_hash,round=round_index,
            actor_scope='readout.residual only',frozen_core_verified=True,coordinates='training reward only',teacher=False,
            std=std.tolist(),gamma=.995,lam=.95,clip=.2,seed=args.seed,anchor_weight=args.anchor_weight,
            optimizer='Adam',critic='separate training-only checkpoint',last_update=stats)
        path=args.out/'latest.pt';save_checkpoint(path,model,actor_opt,[],age)
        torch.save(dict(model=critic.state_dict(),optimizer=critic_opt.state_dict(),round=round_index,
            torch_rng=torch.get_rng_state(),config=vars(args)|dict(source=str(args.source),out=str(args.out))),args.out/'critic.pt')
        write(args.out/'collection.json',collection)
        print(json.dumps(dict(phase='ppo_update',round=round_index,**stats)),flush=True)
        if (round_index+1)%args.evaluate_every==0 or round_index+1==args.rounds:
            old=evaluate(model,[(c,(0.,0.,0.),'old') for c in old_cases]);new=evaluate(model,dev)
            complex_reports=[r for r in new['reports'] if r['family']=='complex']
            passed=old['successes']==7 and old['contacts']==0 and all(r['contacts']==0 for r in complex_reports) and sum(r['success'] for r in complex_reports)>=math.ceil(.9*len(complex_reports))
            summary=dict(round=round_index,old=old['successes'],old_contacts=old['contacts'],new=new['successes'],new_contacts=new['contacts'],
                complex=sum(r['success'] for r in complex_reports),complex_episodes=len(complex_reports),development_gate=passed)
            results.append(summary);write(args.out/f'candidate-{round_index}-development.json',dict(old=old,new=new,summary=summary))
            save_checkpoint(args.out/f'candidate-{round_index}.pt',model,actor_opt,[],age)
            write(args.out/'result.json',dict(candidates=results,deployed=False,final_not_used=True))
            print(json.dumps(dict(phase='ppo_candidate',**summary)),flush=True)
            if passed:break
    assert hashlib.sha256(args.source.read_bytes()).hexdigest()==source_hash


if __name__=='__main__':main()
