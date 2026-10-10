"""Collect actual mirrored sensor histories using reflected training commands."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.sim.scenarios import Scenario
from fly_brain_rc.imitation.curriculum import validation_cases
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from fly_brain_rc.imitation.model import load_checkpoint
from scripts.train_sequential import episode
from scripts.improve_navigation import write


def mirror_case(case):
    return Scenario(case.name+'-physical-mirror',tuple(((x,-y),color) for (x,y),color in case.obstacles),(case.goal[0],-case.goal[1]))


def mirror_pose(pose):return (pose[0],-pose[1],-pose[2])


def mirror_actions(y):
    if y.ndim!=2 or y.shape[1]!=2 or not len(y) or not torch.isfinite(y).all():raise ValueError('finite nonempty paired commands required')
    return y*y.new_tensor([1.,-1.])


class PlaybackTeacher:
    def __init__(self,y):self.y=y;self.index=0
    def action(self):
        command=self.y[min(self.index,len(self.y)-1)];self.index+=1
        return tuple(command.tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','replay','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'run.py').write_text(Path(__file__).read_text());torch.set_num_threads(2)
    payload=torch.load(args.source,weights_only=True);model,*_=load_checkpoint(args.source,payload['graph'])
    saved=torch.load(args.replay,weights_only=True);original=saved['episodes']
    known={c.name:(c,p) for seed in range(31000,31004) for c,p,t in mixed_cases(seed,'train',1)}
    known.update({c.name:(c,(0.,0.,0.)) for c in validation_cases()})
    anchors=[(e['x'],e['y']) for e in original if e['tier']=='simple']
    torch.save(dict(episodes=original,anchors=anchors),args.out/'control.pt')
    augmented=list(original);reports=[]
    for i,e in enumerate(original):
        case,pose=known[e['case']];pose=e.get('pose',pose)
        mc,mp=mirror_case(case),mirror_pose(pose);commands=mirror_actions(e['y'])
        x,y,r=episode(model,mc,mp,beta=1.,teacher_factory=lambda env:PlaybackTeacher(commands))
        r.update(original_index=i,original_case=case.name,tier=e['tier']);reports.append(r)
        # Stable arrival and zero contacts validate reflected playback physically.
        if r['success'] and r['contacts']==0:
            augmented.append(dict(x=x,y=y,tier=e['tier'],case=mc.name,beta=1.,pose=mp,mirror_of=i))
        torch.save(dict(episodes=augmented,anchors=anchors,collection=reports,
            source_replay=str(args.replay.resolve()),source_replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest()),args.out/'augmented.pt')
        write(args.out/'result.json',dict(original=len(original),attempted=len(reports),accepted=len(augmented)-len(original),reports=reports))
    if len(augmented)!=2*len(original):raise RuntimeError('incomplete mirror replay; inspect rejected physical realizations before A/B')


if __name__=='__main__':main()
