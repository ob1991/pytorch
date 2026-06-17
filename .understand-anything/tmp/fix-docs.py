#!/usr/bin/env python3
"""Replace generic documents edges with precise per-directory mappings."""
import json, os

KG = "/home/bluce/pytorch/.understand-anything/knowledge-graph.json"

with open(KG) as f:
    graph = json.load(f)

nodes = graph["nodes"]
edges = graph["edges"]
node_map = {n["id"]: n for n in nodes}

# Remove ALL existing documents edges
old_doc_edges = [e for e in edges if e["type"] == "documents"]
for e in old_doc_edges:
    edges.remove(e)
print(f"Removed {len(old_doc_edges)} generic documents edges")

# Build precise mappings: for each doc, find the most specific __init__.py it documents
doc_nodes = [n for n in nodes if n["type"] == "document"]
added = 0
skipped_license = 0

for doc in doc_nodes:
    fp = doc.get("filePath", doc.get("id", ""))
    doc_id = doc["id"]
    name = os.path.basename(fp).lower()
    doc_dir = os.path.dirname(fp)

    # Skip license files
    if name == "license.txt":
        skipped_license += 1
        continue

    targets = []

    if name == "claude.md":
        # Dynamo CLAUDE.md
        targets = ["file:torch/_dynamo/__init__.py", "file:torch/_dynamo/eval_frame.py"]

    elif name == "cuda_multiprocessing.md":
        targets = ["file:torch/multiprocessing/__init__.py", "file:torch/cuda/__init__.py"]

    elif name == "pattern.md":
        targets = ["file:torch/ao/quantization/__init__.py"]

    elif name == "mangling.md":
        targets = ["file:torch/package/__init__.py"]

    elif name == "contributing.md":
        # CONTRIBUTING.md in torch/distributed/
        targets = ["file:torch/distributed/__init__.py"]

    elif name == "readme.md":
        # Map README to the __init__.py in its own directory
        # Try direct __init__.py in the same dir
        candidate = f"file:{doc_dir}/__init__.py"
        if candidate in node_map:
            targets = [candidate]
        else:
            # Try parent dir
            parent_candidate = f"file:{os.path.dirname(doc_dir)}/__init__.py"
            if parent_candidate in node_map:
                targets = [parent_candidate]

    for t in targets:
        if t in node_map:
            edges.append({
                "source": doc_id,
                "target": t,
                "type": "documents",
                "direction": "forward",
                "weight": 0.5,
            })
            added += 1

graph["edges"] = edges

with open(KG, "w", encoding="utf-8") as f:
    json.dump(graph, f, indent=2, ensure_ascii=False)

# Stats
types = {}
for n in nodes:
    t = n.get('type', '?')
    types[t] = types.get(t, 0) + 1
edge_types = {}
for e in edges:
    t = e.get('type', '?')
    edge_types[t] = edge_types.get(t, 0) + 1

print(f"Added {added} precise documents edges")
print(f"Skipped {skipped_license} license files")
print(f"\n{'='*40}")
print(f"最终统计:")
print(f"  节点: {len(nodes)}")
for t, c in sorted(types.items(), key=lambda x: -x[1]):
    print(f"    {t}: {c}")
print(f"  边: {len(edges)}")
for t, c in sorted(edge_types.items(), key=lambda x: -x[1]):
    print(f"    {t}: {c}")
