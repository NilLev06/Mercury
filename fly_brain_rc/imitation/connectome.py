"""CPU-budgeted induced subgraph of published anatomical connections.

Selection is computational, not a claimed biological sensorimotor circuit.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import networkx as nx

ARCHIVE_SHA256 = '49d19491c1df4b8f02af0e285b71dc171f47488a93ee5b91be8ee4e280bee9f1'
SOURCE = 'https://huggingface.co/datasets/benquan1/fly-gym-trained-policies'


def checksum(path):
    digest = hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    with open(path,newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'pre_root_id','post_root_id','syn_count'} <= set(reader.fieldnames or ()):
            raise ValueError('expected FlyWire connection columns')
        for row in reader:
            count = int(row['syn_count'])
            if count >= 5:
                yield row['pre_root_id'],row['post_root_id'],count


def extract_graph(path,limit=256,*,fixture=False):
    if type(limit) is not int or limit < 3 or limit > 1024:
        raise ValueError('CPU experiment requires 3..1024 selected neurons')
    # First select strongly connected high-degree candidates. Two streaming
    # passes avoid constructing the multi-million-edge whole graph in RAM.
    degrees = Counter()
    for pre,post,count in rows(path):
        degrees[pre] += count
        degrees[post] += count
    candidates = set(sorted(degrees,key=lambda n:(-degrees[n],n))[:limit*4])
    edges = Counter()
    for pre,post,count in rows(path):
        if pre in candidates and post in candidates:
            edges[pre,post] += count
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    components = sorted(nx.strongly_connected_components(graph),key=lambda c:(-len(c),min(c)))
    if not components or len(components[0]) < 3:
        raise ValueError('no usable strongly connected anatomical component')
    # BFS within the largest SCC chooses a compact deterministic subgraph;
    # extract its largest SCC again, since cutting can break connectivity.
    component = components[0]
    root = min(component,key=lambda n:(-degrees[n],n))
    selected,queue = set(),[root]
    for node in queue:
        if node in selected:
            continue
        selected.add(node)
        if len(selected)>=limit:
            break
        queue.extend(sorted((set(graph.successors(node))|set(graph.predecessors(node)))&component-selected))
    induced = graph.subgraph(selected)
    selected = max(nx.strongly_connected_components(induced),key=len)
    if len(selected)<3:
        raise ValueError('selected component too small')
    ids = sorted(selected)
    index = {value:i for i,value in enumerate(ids)}
    connections = sorted([index[pre],index[post],count] for (pre,post),count in edges.items()
                         if pre in selected and post in selected)
    return {'ids':ids,'edges':connections,'manifest':{
        'source': 'synthetic test fixture' if fixture else SOURCE,
        'dataset':'FlyWire FAFB v783' if not fixture else 'fixture',
        'license':'CC BY-NC 4.0' if not fixture else 'test-only',
        'csv_sha256':checksum(path),'fixture':fixture,
        'selection':'weighted degree candidates, BFS, largest induced SCC',
        'neurons':len(ids),'edges':len(connections)}}


def prepare_graph(directory):
    directory = Path(directory)
    cache = directory/'subgraph-256.json'
    if cache.exists():
        graph = json.loads(cache.read_text())
        if graph['manifest'].get('fixture'):
            raise ValueError('test graph cannot be used for the experiment')
        return graph
    archive = directory/'flywire-v783.7z'
    if not archive.exists() or checksum(archive) != ARCHIVE_SHA256:
        raise ValueError('Download the checksum-verified FlyWire archive; see docs/2026-10-06-connectome-imitation.md')
    csv_path = directory/'raw/drosophila adult connectome/connections_princeton.csv'
    if not csv_path.exists():
        import py7zr
        with py7zr.SevenZipFile(archive) as archive_file:
            archive_file.extract(path=directory/'raw',targets=[
                'drosophila adult connectome/connections_princeton.csv',
                'drosophila adult connectome/data source.txt'])
    graph = extract_graph(csv_path)
    graph['manifest']['archive_sha256'] = ARCHIVE_SHA256
    cache.write_text(json.dumps(graph))
    return graph
