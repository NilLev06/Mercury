"""Optional hemisphere-specific physiology hypotheses, never rewired anatomy.

Motivated by Duan et al. (NeurIPS 2025), Appendix A.1. This is NOT an exact
replication: we retain FlyWire counts rather than their hemibrain preprocessing.
"""
import copy
import csv
from pathlib import Path

from .heading_columns import digest,selected_neurons


def split_hemisphere_parameters(anatomy,parameters,annotations_path):
    """Join explicit side annotations by root ID; do not infer side from phase.

    Initially duplicates the supplied shared parameters, preserving all dynamics.
    Side-specific gains/biases can subsequently be calibrated independently.
    """
    if 'cell_classes' in parameters:
        raise ValueError('parameters already have cell classes')
    neurons=selected_neurons(anatomy)
    sides={}
    path=Path(annotations_path)
    with path.open() as stream:
        reader=csv.DictReader(stream,delimiter='\t')
        if not {'root_id','side'}<=set(reader.fieldnames or ()):
            raise ValueError('root_id and side annotations required')
        for row in reader:
            root=row['root_id']
            if root not in neurons:continue
            if root in sides or row['side'] not in ('left','right'):
                raise ValueError('duplicate or unresolved selected side annotation')
            sides[root]=row['side']
    if set(sides)!=set(neurons):
        raise ValueError('missing selected side annotation')
    classes={root:f"{n['cell_type']}:{sides[root]}" for root,n in neurons.items()}
    base={classes[root]:n['cell_type'] for root,n in neurons.items()}
    result=copy.deepcopy(parameters)
    result['gain']={post:{pre:parameters['gain'][base[post]][base[pre]] for pre in base}
                    for post in base}
    for field in ('bias','tau'):
        result[field]={kind:parameters[field][original] for kind,original in base.items()}
    result['cell_classes']=classes
    result['lateralization_source']=dict(path=str(path),sha256=digest(path),
                                        join='root_id',field='side',rewired=False)
    return result
