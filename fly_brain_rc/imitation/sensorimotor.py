"""Directed anatomical corridor selection, not a reconstructed living brain.

Input/output assignments are explicit experimental interfaces. No new edges
are invented and neuron count is never reduced by silently cutting a path.
"""
import argparse
from array import array
from collections import deque
import csv
import json
from pathlib import Path

import numpy as np

from .connectome import checksum


def _distances(pre, post, size, starts, limit):
    order = np.argsort(pre,kind='stable')
    neighbors = post[order]
    offsets = np.r_[0,np.cumsum(np.bincount(pre,minlength=size))]
    distances = np.full(size,np.inf)
    distances[starts] = 0
    queue = deque(starts)
    while queue:
        node = queue.popleft()
        distance = distances[node]+1
        if distance > limit:
            continue
        for other in neighbors[offsets[node]:offsets[node+1]]:
            if distances[other] == np.inf:
                distances[other] = distance
                queue.append(int(other))
    return distances


def select_paths(annotations, edges, inputs, outputs, *, max_hops=6,
                 max_neurons=20000, min_synapses=5, require_all_inputs=True):
    if (type(max_hops) is not int or not 1 <= max_hops <= 12
            or type(max_neurons) is not int or max_neurons < 1
            or type(min_synapses) is not int or min_synapses < 1):
        raise ValueError('positive bounded path and budget settings required')
    ids = sorted(annotations)
    index = {root: i for i, root in enumerate(ids)}
    source_ids = {root for group in inputs.values() for root in group}
    output_ids = set(outputs)
    if (not inputs or any(not group for group in inputs.values()) or not output_ids
            or not source_ids <= index.keys() or not output_ids <= index.keys()
            or source_ids & output_ids):
        raise ValueError('nonempty disjoint annotated input/output IDs required')
    pre, post, counts = array('i'), array('i'), array('d')
    for a, b, count in edges:
        if not np.isfinite(count) or count <= 0 or int(count) != count:
            raise ValueError('positive integer synapse counts required')
        if a in index and b in index:
            pre.append(index[a]); post.append(index[b]); counts.append(count)
    pre,post,counts = np.asarray(pre),np.asarray(post),np.asarray(counts)
    order = np.lexsort((post,pre))
    pre,post,counts = pre[order],post[order],counts[order]
    # Aggregate regional rows BEFORE applying the neuron-pair threshold.
    if len(pre):
        starts = np.flatnonzero(np.r_[True,(pre[1:]!=pre[:-1])|(post[1:]!=post[:-1])])
        pre,post,counts = pre[starts],post[starts],np.add.reduceat(counts,starts)
    keep = counts >= min_synapses
    pre,post,counts = pre[keep],post[keep],counts[keep]
    sources = sorted(index[root] for root in source_ids)
    sinks = sorted(index[root] for root in output_ids)
    forward = _distances(pre,post,len(ids),sources,max_hops)
    backward = _distances(post,pre,len(ids),sinks,max_hops)
    selected = np.flatnonzero(forward + backward <= max_hops)
    selected_set = set(selected.tolist())
    if (not set(sinks) <= selected_set
            or require_all_inputs and not set(sources) <= selected_set
            or any(not any(index[root] in selected_set for root in group)
                   for group in inputs.values())):
        raise ValueError('unreachable input/output within directed hop bound')
    if len(selected) > max_neurons:
        raise ValueError(f'neuron budget exceeded: {len(selected)} > {max_neurons}')
    remap = {old: new for new, old in enumerate(selected)}
    keep = np.isin(pre,selected)&np.isin(post,selected)
    return dict(ids=[ids[i] for i in selected],
                edges=sorted([remap[int(a)],remap[int(b)],int(c)] for a,b,c in
                             zip(pre[keep],post[keep],counts[keep])),
                neurons=[annotations[ids[i]] for i in selected],
                inputs={name:sorted(remap[index[root]] for root in set(group)
                                    if index[root] in selected_set)
                        for name,group in inputs.items()},
                excluded_inputs={name:sorted(root for root in set(group)
                                             if index[root] not in selected_set)
                                 for name,group in inputs.items()},
                outputs=sorted(remap[i] for i in sinks),
                selection=dict(max_hops=max_hops,min_synapses=min_synapses,
                               rule='directed distance from input + distance to output <= max_hops'),
                vehicle_integration_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--connections',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-hops',type=int,default=6)
    parser.add_argument('--max-neurons',type=int,default=20000)
    parser.add_argument('--allow-partial-inputs',action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('new output path required')
    with args.annotations.open() as handle:
        annotations = {r['root_id']: {key:r[key] for key in
            ('cell_type','hemibrain_type','side','flow','top_nt','cell_class')}
            for r in csv.DictReader(handle,delimiter='\t')}
    # Camera-derived object features enter visual projection cells for this
    # first pilot, NOT photoreceptors. Touch uses annotated mechanosensory
    # afferents. This is an engineered mapping, not natural sensory physiology.
    inputs = {}
    for kind in ('LC10a','LC4','LPLC2','BM_InOm'):
        for side in ('left','right'):
            inputs[f'{kind}:{side}'] = [root for root,r in annotations.items()
                if kind in (r['cell_type'],r['hemibrain_type']) and r['side']==side]
    outputs = [root for root,r in annotations.items() if r['flow']=='efferent'
        and any(kind in (r['cell_type'],r['hemibrain_type'])
                for kind in ('DNa02','DNa03','DNp09'))]
    with args.connections.open() as handle:
        edges = ((r['pre_root_id'],r['post_root_id'],int(r['syn_count']))
                 for r in csv.DictReader(handle))
        graph = select_paths(annotations,edges,inputs,outputs,
                             max_hops=args.max_hops,max_neurons=args.max_neurons,
                             require_all_inputs=not args.allow_partial_inputs)
    graph['provenance'] = dict(annotations_sha256=checksum(args.annotations),
                              connections_sha256=checksum(args.connections),
                              dataset='FlyWire FAFB v783',
                              input_mapping='engineered visual-projection and tactile pilot')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(graph,handle)
    print(json.dumps(dict(neurons=len(graph['ids']),edges=len(graph['edges']),
                          inputs={k:len(v) for k,v in graph['inputs'].items()},
                          excluded_inputs={k:len(v) for k,v in graph['excluded_inputs'].items()},
                          outputs=len(graph['outputs']),out=str(args.out))))


if __name__ == '__main__':
    main()
