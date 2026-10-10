"""Inference-only recurrence-weight ablation; never writes a brain checkpoint."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.model import load_checkpoint
from fly_brain_rc.imitation.curriculum import validation_cases
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from scripts.train_sequential import evaluate
from scripts.improve_navigation import write
p=Path('artifacts/imitation/experiments/frozen-navigation-final-20261009/selected.pt');sha=hashlib.sha256(p.read_bytes()).hexdigest();raw=torch.load(p,weights_only=True)
torch.set_num_threads(2);m,*_=load_checkpoint(p,raw['graph'])
with torch.no_grad():m.recurrent.zero_()
r=evaluate(m,[(c,(0.,0.,0.),'old') for c in validation_cases()]+[c for c in mixed_cases(20001,'selection',1) if c[2]=='complex'])
write(p.parent/'recurrence-ablation.json',dict(source_sha256=sha,zero_recurrent_weight_in_memory=True,reports=r,
 conclusion='tests dependence on trained recurrent weights, not biological fidelity or temporal memory alone',source_unchanged=hashlib.sha256(p.read_bytes()).hexdigest()==sha))
print(json.dumps(dict(phase='ablation_complete',successes=r['successes'],episodes=r['episodes'],contacts=r['contacts'])),flush=True)
