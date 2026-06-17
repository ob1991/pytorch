#!/usr/bin/env python3
"""Scan docs/ and add document nodes + edges to the knowledge graph."""
import json, os, re, glob

KG = "/home/bluce/pytorch/.understand-anything/knowledge-graph.json"

with open(KG) as f:
    graph = json.load(f)

nodes = graph["nodes"]
edges = graph["edges"]
node_map = {n["id"]: n for n in nodes}

# Existing doc node IDs for dedup
existing_doc_ids = {n["id"] for n in nodes if n["type"] == "document"}

# ── 1. Scan docs/ for documentation files ──
doc_files = []
for root, dirs, files in os.walk("docs/"):
    # Skip build artifacts and generated directories
    skip_dirs = {"build", "_build", "venv", "node_modules", "generated", "__pycache__"}
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for f in files:
        if f.endswith((".md", ".rst")):
            fp = os.path.join(root, f)
            with open(fp, "rb") as fh:
                lines = sum(1 for _ in fh)
            doc_files.append((fp, lines))

print(f"Found {len(doc_files)} documentation files in docs/")

# ── 2. Build mapping: doc → code targets ──
# Convention: docs/source/<name>.rst/md → torch/<name> module

def doc_to_code_targets(doc_path):
    """Map a docs/ file path to the torch code it documents."""
    rel = doc_path[len("docs/"):]
    name = os.path.splitext(os.path.basename(rel))[0]

    # Manual mappings for known docs
    manual_map = {
        "torch":               ["file:torch/__init__.py"],
        "tensors":             ["file:torch/_tensor.py", "file:torch/__init__.py"],
        "autograd":            ["file:torch/autograd/__init__.py"],
        "cuda":                ["file:torch/cuda/__init__.py"],
        "nn":                  ["file:torch/nn/__init__.py"],
        "nn.attention":        ["file:torch/nn/__init__.py"],
        "nn.attention.experimental": ["file:torch/nn/__init__.py"],
        "nn.attention.varlen": ["file:torch/nn/__init__.py"],
        "fx":                  ["file:torch/fx/__init__.py"],
        "fx.experimental":     ["file:torch/fx/experimental/__init__.py"],
        "optim":               ["file:torch/optim/__init__.py"],
        "data":                ["file:torch/utils/data/__init__.py"],
        "distributed":         ["file:torch/distributed/__init__.py"],
        "distributed.tensor":  ["file:torch/distributed/tensor/__init__.py"],
        "distributed.pipelining": ["file:torch/distributed/pipelining/__init__.py"],
        "distributed.optim":   ["file:torch/distributed/optim/__init__.py"],
        "distributed.checkpoint": ["file:torch/distributed/checkpoint/__init__.py"],
        "distributed.fsdp":    ["file:torch/distributed/fsdp/__init__.py"],
        "distributed.fsdp.fully_shard": ["file:torch/distributed/_composable/fsdp.py"],
        "ddp_comm_hooks":      ["file:torch/distributed/algorithms/ddp_comm_hooks/__init__.py"],
        "onnx":                ["file:torch/onnx/__init__.py"],
        "onnx_testing":        ["file:torch/onnx/__init__.py"],
        "jit":                 ["file:torch/jit/__init__.py"],
        "jit_utils":           ["file:torch/jit/_script.py"],
        "quantization":        ["file:torch/ao/quantization/__init__.py"],
        "quantization-support":["file:torch/ao/quantization/fx/__init__.py"],
        "profiler":            ["file:torch/profiler/__init__.py"],
        "tensorboard":         ["file:torch/utils/tensorboard/__init__.py"],
        "rpc":                 ["file:torch/distributed/rpc/__init__.py"],
        "distributions":       ["file:torch/distributions/__init__.py"],
        "sparse":              ["file:torch/sparse/__init__.py"],
        "linalg":              ["file:torch/linalg/__init__.py"],
        "fft":                 ["file:torch/fft/__init__.py"],
        "special":             ["file:torch/special/__init__.py"],
        "nested":              ["file:torch/nested/__init__.py"],
        "masked":              ["file:torch/_masked/__init__.py"],
        "utils":               ["file:torch/utils/__init__.py"],
        "benchmark_utils":     ["file:torch/utils/benchmark/__init__.py"],
        "checkpoint":          ["file:torch/utils/checkpoint.py"],
        "dlpack":              ["file:torch/dlpack/capsule.py"],
        "complex_numbers":     ["file:torch/_tensor.py"],
        "named_tensor":        ["file:torch/_tensor.py"],
        "tensor_attributes":   ["file:torch/_tensor.py"],
        "library":             ["file:torch/library.py"],
        "monitor":             ["file:torch/monitor/__init__.py"],
        "backends":            ["file:torch/backends/__init__.py"],
        "meta":                ["file:torch/_meta_registrations.py"],
        "mps":                 ["file:torch/mps/__init__.py"],
        "mps_environment_variables": ["file:torch/mps/__init__.py"],
        "xpu":                 ["file:torch/xpu/__init__.py"],
        "mtia":                ["file:torch/mtia/__init__.py"],
        "mtia.mtia_graph":     ["file:torch/mtia/__init__.py"],
        "mtia.memory":         ["file:torch/mtia/__init__.py"],
        "symmetric_memory":    ["file:torch/_symmetric_memory/__init__.py"],
        "cpu":                 ["file:torch/cpu/__init__.py"],
        "accelerator":         ["file:torch/accelerator/__init__.py"],
        "func.api":            ["file:torch/_functorch/__init__.py"],
        "func.batch_norm":     ["file:torch/_functorch/__init__.py"],
        "func.ux_limitations": ["file:torch/_functorch/__init__.py"],
        "torch.compiler_api":  ["file:torch/_dynamo/__init__.py", "file:torch/_inductor/__init__.py"],
        "mobile_optimizer":    ["file:torch/utils/mobile_optimizer.py"],
        "futures":             ["file:torch/futures/__init__.py"],
        "cpp_index":           ["file:torch/csrc/__init__.pyi.in"],
        "cpp":                 ["file:torch/csrc/__init__.pyi.in"],
    }

    if name in manual_map:
        result = manual_map[name]
        # Filter to targets that actually exist, fall back to torch/__init__.py
        valid = [t for t in result if t in node_map]
        if valid:
            return valid
        return ["file:torch/__init__.py"]

    # Auto-map: docs/source/foo.bar.md → torch/foo/bar/__init__.py
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        sub = parts[:i]
        t = f"file:torch/{'/'.join(sub)}/__init__.py"
        if t in node_map:
            return [t]
    # Try torch/ prefix
    for i in range(len(parts), 0, -1):
        sub = parts[:i]
        t = f"file:torch/{'/'.join(sub)}.py"
        if t in node_map:
            return [t]

    # Fallback to parent directory's __init__.py
    rel_no_ext = os.path.splitext(rel)[0]
    parts = rel_no_ext.split("/")
    for i in range(len(parts), 0, -1):
        sub = parts[:i]
        t = f"file:torch/{'/'.join(sub)}/__init__.py"
        if t in node_map:
            return [t]
        t2 = f"file:{'/'.join(sub)}/__init__.py"
        if t2 in node_map:
            return [t2]
    # Ultimate fallback: torch/__init__.py
    return ["file:torch/__init__.py"]

# ── 3. Generate document nodes and edges ──
new_nodes = []
new_edges = []
existing_doc_edges = set()
for e in edges:
    if e["type"] == "documents":
        existing_doc_edges.add((e["source"], e["target"]))

tag = ["documentation"]

for fp, line_count in doc_files:
    doc_id = f"document:{fp}"
    if doc_id in existing_doc_ids:
        continue

    rel = fp[len("docs/"):]
    name = os.path.splitext(os.path.basename(rel))[0]
    parent = rel.split("/")[0]  # "source" or "cpp" etc.

    summary = f"{os.path.basename(fp)} — PyTorch {name} 文档，{line_count} 行。"

    node = {
        "id": doc_id,
        "type": "document",
        "name": os.path.basename(fp),
        "filePath": fp,
        "summary": summary,
        "tags": tag,
        "complexity": "simple",
    }
    new_nodes.append(node)
    existing_doc_ids.add(doc_id)

    # Add documents edges
    targets = doc_to_code_targets(fp)
    for t in targets:
        if t in node_map and (doc_id, t) not in existing_doc_edges:
            new_edges.append({
                "source": doc_id,
                "target": t,
                "type": "documents",
                "direction": "forward",
                "weight": 0.5,
            })
            existing_doc_edges.add((doc_id, t))

print(f"  New document nodes: {len(new_nodes)}")
print(f"  New documents edges: {len(new_edges)}")

# ── 4. Merge into graph ──
nodes.extend(new_nodes)
edges.extend(new_edges)

graph["nodes"] = nodes
graph["edges"] = edges

with open(KG, "w", encoding="utf-8") as f:
    json.dump(graph, f, indent=2, ensure_ascii=False)

# ── 5. Stats ──
types = {}
for n in nodes:
    t = n.get('type', '?')
    types[t] = types.get(t, 0) + 1
edge_types = {}
for e in edges:
    t = e.get('type', '?')
    edge_types[t] = edge_types.get(t, 0) + 1

doc_count = len([n for n in nodes if n["type"] == "document"])
print(f"\n{'='*40}")
print(f"最终统计:")
print(f"  节点: {len(nodes)} (文档: {doc_count})")
for t, c in sorted(types.items(), key=lambda x: -x[1]):
    print(f"    {t}: {c}")
print(f"  边: {len(edges)}")
for t, c in sorted(edge_types.items(), key=lambda x: -x[1]):
    print(f"    {t}: {c}")
