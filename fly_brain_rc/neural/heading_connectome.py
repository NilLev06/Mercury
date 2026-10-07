"""Anatomical heading-circuit extraction, not a validated dynamics model.

Keeps source direction, synapse counts, regions and transmitter labels.
No arbitrary graph-size selection, invented sectors or inferred signs.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


HEADING_TYPES = ('EPG','PEN_a/PEN1','PEN_b/PEN2','PEG','Delta7')


def _sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path,required):
    with open(path,newline='') as stream:
        reader=csv.DictReader(stream)
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f'missing required CSV columns: {path}')
        yield from reader


def extract_heading_circuit(cells_path,connections_path):
    neurons={}
    for row in _rows(cells_path,{'root_id','primary_type'}):
        kind=row['primary_type'].strip()
        if kind not in HEADING_TYPES:
            continue
        root=row['root_id'].strip()
        if not root.isdigit() or root in neurons:
            raise ValueError('invalid or duplicate selected root_id')
        neurons[root]={'root_id':root,'cell_type':kind}
    counts=Counter(n['cell_type'] for n in neurons.values())
    missing=set(HEADING_TYPES)-counts.keys()
    if missing:
        raise ValueError(f'missing heading populations: {sorted(missing)}')
    connections=[]
    required={'pre_root_id','post_root_id','neuropil','syn_count','nt_type'}
    for row in _rows(connections_path,required):
        pre,post=row['pre_root_id'].strip(),row['post_root_id'].strip()
        if pre not in neurons or post not in neurons:
            continue
        count=int(row['syn_count'])
        if count<0:
            raise ValueError('negative synapse count')
        if count:
            connections.append(dict(pre=pre,post=post,neuropil=row['neuropil'],
                                    syn_count=count,nt_type=row['nt_type']))
    if not connections:
        raise ValueError('no internal heading connections')
    return dict(version=1,functional_model=False,
                neurons=[neurons[n] for n in sorted(neurons,key=int)],
                connections=connections,type_counts=dict(sorted(counts.items())),
                sources=dict(cells_path=str(cells_path),connections_path=str(connections_path),
                             cells_sha256=_sha256(cells_path),
                             connections_sha256=_sha256(connections_path)),
                limitations=['Type-selected induced subgraph, not the complete heading circuit.',
                             'EB/PB sector assignments are absent; never infer them from root IDs.',
                             'Transmitter labels alone do not establish postsynaptic effect.',
                             'Dynamics, visual inputs and turn inputs are not implemented.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells',type=Path,required=True)
    parser.add_argument('--connections',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=extract_heading_circuit(args.cells,args.connections)
    # Avoid accidental destruction of an earlier anatomical report.
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(dict(type_counts=result['type_counts'],
                         neurons=len(result['neurons']),rows=len(result['connections']),
                         synapses=sum(e['syn_count'] for e in result['connections']))))


if __name__=='__main__':
    main()
