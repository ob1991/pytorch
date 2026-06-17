#!/usr/bin/env python3
"""Analyze assembled graph for review."""
import json, sys
from collections import Counter

with open('/home/bluce/pytorch/.understand-anything/intermediate/assembled-graph.json') as f:
    graph = json.load(f)
with open('/home/bluce/pytorch/.understand-anything/intermediate/scan-result.json') as f:
    scan = json.load(f)

nodes = graph['nodes']
edges = graph['edges']
import_map = scan.get('importMap', {})

print(f"Nodes: {len(nodes)}")
print(f"Edges: {len(edges)}")
print(f"Import map entries: {len(import_map)}")

# Node types
type_counts = Counter(n.get('type') for n in nodes)
print(f"Node types: {dict(type_counts)}")

# ID prefixes
prefix_counts = Counter(n['id'].split(':')[0] if ':' in n.get('id','') else 'no-prefix' for n in nodes)
print(f"ID prefixes: {dict(prefix_counts)}")

# Edge types
edge_type_counts = Counter(e.get('type') for e in edges)
print(f"Edge types: {dict(edge_type_counts)}")

# Collect file nodes (by path)
file_nodes = set()
for n in nodes:
    if n['type'] == 'file' and ':' in n.get('id', ''):
        file_nodes.add(n['id'].split(':', 1)[1])
    elif n['type'] == 'file':
        file_nodes.add(n.get('id', ''))

print(f"File nodes count: {len(file_nodes)}")

# Collect all imports edges
import_edges = set()
for e in edges:
    if e.get('type') == 'imports':
        src = e['source'].split(':', 1)[1] if ':' in e['source'] else e['source']
        tgt = e['target'].split(':', 1)[1] if ':' in e['target'] else e['target']
        import_edges.add((src, tgt))

print(f"Import edges count: {len(import_edges)}")

# Find missing import edges
missing_edges = []
for src_file, targets in import_map.items():
    if src_file not in file_nodes:
        continue
    for tgt_file in targets:
        if tgt_file in file_nodes and tgt_file != src_file:
            if (src_file, tgt_file) not in import_edges:
                missing_edges.append((src_file, tgt_file))

print(f"Missing import edges (both files in graph, no edge): {len(missing_edges)}")
if missing_edges:
    print("Sample missing edges (first 50):")
    for src, tgt in missing_edges[:50]:
        print(f"  {src} -> {tgt}")

# Also check: how many importMap entries reference files not in the graph
missing_targets = set()
present_targets = set()
for src_file, targets in import_map.items():
    for tgt_file in targets:
        if tgt_file in file_nodes:
            present_targets.add(tgt_file)
        else:
            missing_targets.add(tgt_file)
print(f"\nTarget files in importMap that ARE in graph: {len(present_targets)}")
print(f"Target files in importMap NOT in graph: {len(missing_targets)}")
if missing_targets:
    print("Sample missing targets (first 30):")
    for t in sorted(list(missing_targets))[:30]:
        print(f"  {t}")

# Check unknown node types
unknown_types = [n.get('type') for n in nodes if n.get('type') not in ('file','function','class','document','config')]
print(f"\nUnknown node types: {Counter(unknown_types)}")

# Check unknown complexity values
valid_complexity = {'simple','moderate','complex'}
bad_complexity = [n.get('complexity') for n in nodes if n.get('complexity') not in valid_complexity]
print(f"Unknown complexity values: {Counter(bad_complexity)}")

# Check nodes without ID
no_id = [n for n in nodes if 'id' not in n]
print(f"Nodes without id: {len(no_id)}")
if no_id:
    print("Sample:", no_id[:3])

# Check dangling edges (edges referencing nodes not in graph)
all_node_ids = set(n.get('id') for n in nodes)
dangling_src = [e for e in edges if e.get('source') not in all_node_ids]
dangling_tgt = [e for e in edges if e.get('target') not in all_node_ids]
print(f"\nEdges with dangling source: {len(dangling_src)}")
print(f"Edges with dangling target: {len(dangling_tgt)}")

# Check for no-id nodes in edge refs
print("\n=== Phase 1: Fixed section sanity ===")
print("Fixed: 31 duplicate node IDs removed")
print(f"31 out of {len(nodes)} = {31/len(nodes)*100:.2f}% - reasonable (< 30%)")

print("\n=== Phase 2: Dropped edges analysis ===")
print("Dropped edges due to missing target nodes (from user report): 134")
print(f"Missing target files in importMap that are not in graph: {len(missing_targets)}")
# These would correspond to files like torch/fx/experimental/symbolic_shapes.py outside scope

print("\n=== Phase 3: Cross-batch edge gaps ===")
print(f"Missing import edges: {len(missing_edges)}")
print(f"We will add {len(missing_edges)} missing import edges to the graph")

# Save missing edges for use in fixing script
print("\n=== Output for fixing script ===")
print(f"MISSING_EDGES_COUNT={len(missing_edges)}")
print("MISSING_EDGES_LIST_START")
for src, tgt in missing_edges:
    print(json.dumps({"source": src, "target": tgt}))
print("MISSING_EDGES_LIST_END")
