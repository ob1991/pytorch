#!/usr/bin/env python3
"""Assign docs/ document nodes to appropriate layers and update meta.json."""
import json

KG = "/home/bluce/pytorch/.understand-anything/knowledge-graph.json"

with open(KG) as f:
    graph = json.load(f)

nodes = graph["nodes"]
edges = graph["edges"]
node_map = {n["id"]: n for n in nodes}

# Build doc → layer mapping via documents edges
doc_layer_count = {}  # doc_id → {layer_id: count}
for e in edges:
    if e["type"] != "documents":
        continue
    doc_id = e["source"]
    if not doc_id.startswith("document:docs/"):
        continue
    tgt = e["target"]
    # Find which layer(s) this target belongs to
    for layer in graph["layers"]:
        if tgt in layer["nodeIds"]:
            doc_layer_count.setdefault(doc_id, {}).setdefault(layer["id"], 0)
            doc_layer_count[doc_id][layer["id"]] += 1

# Assign each doc to its best layer
doc_layer = {}  # doc_id → layer_id
for doc_id, counts in doc_layer_count.items():
    best_layer = max(counts, key=counts.get)
    doc_layer[doc_id] = best_layer

# Docs that have no edge-based layer assignment → use path heuristic
unassigned = [n["id"] for n in nodes
              if n["type"] == "document" and n["id"].startswith("document:docs/")
              and n["id"] not in doc_layer]

path_layer_map = {
    "source/autograd": "autograd",
    "source/cuda": "cuda",
    "source/nn": "nn",
    "source/fx": "fx",
    "source/onnx": "onnx",
    "source/jit": "jit",
    "source/quantization": "quantization",
    "source/distributed": "distributed",
    "source/distributions": "core",
    "source/torch": "core",
    "source/tensors": "core",
    "source/data": "utils",
    "source/profiler": "utils",
    "source/optim": "nn",
    "source/sparse": "sparse",
    "source/linalg": "sparse",
    "source/special": "core",
    "source/library": "core",
    "source/backends": "core",
    "source/checkpoint": "utils",
    "source/dlpack": "core",
    "source/benchmark_utils": "utils",
    "source/utils": "utils",
    "source/tensorboard": "utils",
    "source/rpc": "distributed",
    "source/nested": "core",
    "source/meta": "core",
    "source/fft": "sparse",
    "source/func": "functorch",
    "source/monitor": "utils",
    "source/torch.compiler_api": "dynamo",
    "source/mobile_optimizer": "core",
    "source/accelerator": "core",
    "source/amp": "cuda",
    "source/complex_numbers": "core",
    "source/named_tensor": "core",
    "source/tensor_attributes": "core",
    "source/cpp": "core",
    "source/cpp_index": "core",
    "source/cpu": "core",
    "source/xpu": "core",
    "source/mtia": "core",
    "source/mps": "core",
    "source/symmetric_memory": "core",
    "source/masked": "core",
    "source/futures": "distributed",
    "source/community": "core",
}

# Assign unassigned via path heuristic
for doc_id in unassigned:
    rel = doc_id.replace("document:docs/", "")
    assigned = False
    for pattern, layer_name in path_layer_map.items():
        if rel.startswith(pattern):
            layer_id = f"layer:{layer_name}"
            doc_layer[doc_id] = layer_id
            assigned = True
            break
    if not assigned:
        doc_layer[doc_id] = "layer:core"  # default

# Also update edge-assigned docs that conflict with path heuristic
# (already assigned is fine)

# Add doc IDs to layers
layer_map = {l["id"]: l for l in graph["layers"]}
added_count = 0
for doc_id, layer_id in doc_layer.items():
    if layer_id in layer_map and doc_id not in layer_map[layer_id]["nodeIds"]:
        layer_map[layer_id]["nodeIds"].append(doc_id)
        added_count += 1

# Update nodeCount for each layer
for layer in graph["layers"]:
    layer["nodeCount"] = len(layer["nodeIds"])

print(f"Assigned {added_count} doc nodes to layers")
for layer in graph["layers"]:
    print(f"  {layer['id']}: {layer['nodeCount']}")

# Save
with open(KG, "w", encoding="utf-8") as f:
    json.dump(graph, f, indent=2, ensure_ascii=False)

# Update meta.json
import os
META = "/home/bluce/pytorch/.understand-anything/meta.json"
with open(META) as f:
    meta = json.load(f)
meta["nodeCount"] = len(nodes)
meta["edgeCount"] = len(edges)
meta["docCount"] = len([n for n in nodes if n["type"] == "document"])
meta["lastAnalyzedAt"] = "2026-06-03T12:00:00Z"
with open(META, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"\nmeta.json updated: {len(nodes)} nodes, {len(edges)} edges")
