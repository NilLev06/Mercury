"""Development-only selection of static neural parameter blends, no runtime gates."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.model import load_checkpoint,save_checkpoint
from fly_brain_rc.imitation.curriculum import validation_cases
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from scripts.improve_navigation import blend_residual,write
from scripts.train_sequential import evaluate
from scripts.train_neural_ppo import freeze_actor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True);parser.add_argument('--toward',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);(args.out/'run.py').write_text(Path(__file__).read_text());torch.set_num_threads(2)
    a=torch.load(args.base,weights_only=True);b=torch.load(args.toward,weights_only=True);model,*_=load_checkpoint(args.base,a['graph']);freeze_actor(model)
    blocked=next(c for c in validation_cases() if c.name=='blocked_goal');results=[]
    for scale in (.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9):
        model.load_state_dict(blend_residual(a['model'],b['model'],scale))
        probe=evaluate(model,[(blocked,(0.,0.,0.),'old')]);row=dict(scale=scale,blocked=probe)
        if probe['successes']==1 and probe['contacts']==0:
            old=evaluate(model,[(c,(0.,0.,0.),'old') for c in validation_cases()]);row['old']=old
            if old['successes']==7 and old['contacts']==0:
                new=evaluate(model,mixed_cases(20001,'selection',1));row['new']=new
                complex_reports=[r for r in new['reports'] if r['family']=='complex']
                passed=sum(r['success'] for r in complex_reports)==6 and all(r['contacts']==0 for r in complex_reports)
                row['development_gate']=passed
                model.training_metadata['development_parameter_blend']=dict(base=str(args.base.resolve()),toward=str(args.toward.resolve()),
                    base_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest(),toward_sha256=hashlib.sha256(args.toward.read_bytes()).hexdigest(),scale=scale,
                    runtime='single neural checkpoint; no runtime switching or action rules',optimizer_reset=True)
                path=args.out/f'candidate-{scale}.pt';save_checkpoint(path,model,torch.optim.Adam(model.parameters(),lr=3e-4),[],a['iteration'])
                row['checkpoint']=str(path.resolve())
                if passed:
                    write(args.out/'qualified.json',row)
        results.append(row);write(args.out/'result.json',dict(options=results,deployed=False,final_not_used=True))
        print(json.dumps(dict(phase='blend_option',scale=scale,blocked=probe['successes'],old=row.get('old',{}).get('successes'),new=row.get('new',{}).get('successes'),qualified=row.get('development_gate',False))),flush=True)
        if row.get('development_gate'):break


if __name__=='__main__':main()
