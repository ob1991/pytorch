#!/usr/bin/env python3
"""
Enhance the knowledge graph with:
1. documents edges (documentation → code)
2. depends_on edges (known architectural relationships)
3. Better function summaries from extraction data
4. Fix module node ID prefixes (layer: → module:)
5. Add configures edges for config files
"""
import json, os, re

KG = "/home/bluce/pytorch/.understand-anything/knowledge-graph.json"

with open(KG) as f:
    graph = json.load(f)

nodes = graph["nodes"]
edges = graph["edges"]
node_map = {n["id"]: n for n in nodes}

# ── 1. Fix module node IDs ──
print("=== 1. Fixing module node ID prefixes ===")
fixed_count = 0
for n in nodes:
    if n["type"] == "module" and n["id"].startswith("layer:"):
        old_id = n["id"]
        new_id = "module:" + old_id[len("layer:"):]
        n["id"] = new_id
        n["name"] = n.get("name", new_id.split(":")[-1])
        # Update references in edges
        for e in edges:
            if e["source"] == old_id:
                e["source"] = new_id
            if e["target"] == old_id:
                e["target"] = new_id
        node_map[new_id] = n
        if old_id in node_map:
            del node_map[old_id]
        fixed_count += 1
        print(f"  {old_id} → {new_id}")
print(f"  Fixed {fixed_count} module node IDs")

# ── 2. Connect document nodes with documents edges ──
print("\n=== 2. Adding documents edges ===")
doc_nodes = [n for n in nodes if n["type"] == "document"]
existing_doc_targets = set()
for e in edges:
    if e["type"] == "documents":
        existing_doc_targets.add((e["source"], e["target"]))

doc_edges_added = 0
doc_mappings = {
    # README-like docs map to entry points
    re.compile(r".*readme.*", re.I): ["file:torch/__init__.py", "file:torch/_tensor.py"],
}

# Map by filename patterns
for doc in doc_nodes:
    fp = doc.get("filePath", doc.get("id", ""))
    name = os.path.basename(fp).lower()
    targets = []
    if "readme" in name:
        targets = ["file:torch/__init__.py"]
    elif "contributing" in name:
        targets = ["file:torch/__init__.py", "file:torchgen/gen.py"]
    elif "license" in name:
        continue  # skip license files
    elif "cmake" in name.lower():
        targets = ["file:torch/_inductor/__init__.py", "file:torch/cuda/__init__.py"]
    elif "cuda" in fp.lower():
        targets = ["file:torch/cuda/__init__.py"]
    elif "distributed" in fp.lower():
        targets = ["file:torch/distributed/__init__.py"]
    elif "onnx" in fp.lower():
        targets = ["file:torch/onnx/__init__.py"]
    elif "export" in fp.lower():
        targets = ["file:torch/export/__init__.py"]

    for t in targets:
        # Only add if edge doesn't already exist
        ekey = (doc["id"], t)
        if ekey not in existing_doc_targets and t in node_map:
            edges.append({
                "source": doc["id"],
                "target": t,
                "type": "documents",
                "direction": "forward",
                "weight": 0.5,
            })
            existing_doc_targets.add(ekey)
            doc_edges_added += 1

print(f"  Added {doc_edges_added} documents edges")

# ── 3. Add key depends_on edges for well-known architectural dependencies ──
print("\n=== 3. Adding depends_on edges ===")

# Known architectural dependency chains in PyTorch
dep_chains = [
    # Dynamo depends on FX
    ("file:torch/_dynamo/__init__.py", "file:torch/fx/__init__.py"),
    ("file:torch/_dynamo/eval_frame.py", "file:torch/fx/graph.py"),
    ("file:torch/_dynamo/convert_frame.py", "file:torch/fx/graph_module.py"),
    # Inductor depends on Dynamo and FX
    ("file:torch/_inductor/compile_fx.py", "file:torch/_dynamo/__init__.py"),
    ("file:torch/_inductor/graph.py", "file:torch/fx/__init__.py"),
    ("file:torch/_inductor/lowering.py", "file:torch/_dynamo/__init__.py"),
    # Export depends on FX and Dynamo
    ("file:torch/export/exported_program.py", "file:torch/fx/graph.py"),
    ("file:torch/export/dynamic_shapes.py", "file:torch/_dynamo/__init__.py"),
    # CUDA depends on backends
    ("file:torch/cuda/__init__.py", "file:torch/backends/cuda/__init__.py"),
    # Distributed depends on CUDA
    ("file:torch/distributed/distributed_c10d.py", "file:torch/cuda/__init__.py"),
    # Autograd is used by NN
    ("file:torch/nn/__init__.py", "file:torch/autograd/__init__.py"),
    # Torchgen generates code for all
    ("file:torchgen/gen.py", "file:torch/_ops.py"),
    # FX depends on core
    ("file:torch/fx/graph.py", "file:torch/_tensor.py"),
    # Optim depends on NN
    ("file:torch/optim/optimizer.py", "file:torch/nn/__init__.py"),
]

existing_dep_edges = set()
for e in edges:
    if e["type"] == "depends_on":
        existing_dep_edges.add((e["source"], e["target"]))

dep_edges_added = 0
for src, tgt in dep_chains:
    if (src, tgt) not in existing_dep_edges and src in node_map and tgt in node_map:
        edges.append({
            "source": src,
            "target": tgt,
            "type": "depends_on",
            "direction": "forward",
            "weight": 0.6,
        })
        dep_edges_added += 1

print(f"  Added {dep_edges_added} depends_on edges")

# ── 4. Enhance function summaries from extraction data ──
print("\n=== 4. Enhancing function node summaries ===")

# Check if extraction results are available
extraction_dir = "/home/bluce/pytorch/.understand-anything/tmp"
enhanced = 0
for i in range(10):
    fpath = f"{extraction_dir}/ua-file-extract-results-{i}.json"
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        data = json.load(f)
    for result in data.get("results", []):
        fp = result.get("path", "")
        # Enhance function nodes for this file
        for fn in result.get("functions", []):
            name = fn.get("name", "")
            if not name:
                continue
            fn_id = f"function:{fp}:{name}"
            if fn_id in node_map:
                fn_node = node_map[fn_id]
                line_range = fn.get("endLine", 0) - fn.get("startLine", 0)
                params = ", ".join(fn.get("params", [])[:4])
                if params:
                    fn_node["summary"] = f"{name}({params}) — {fp} 中的函数，{line_range} 行代码。"
                else:
                    fn_node["summary"] = f"{name} — {fp} 中的函数，{line_range} 行代码。"
                enhanced += 1

print(f"  Enhanced {enhanced} function node summaries")

# ── 5. Enhance class nodes with method info ──
print("\n=== 5. Enhancing class node summaries ===")
enhanced_classes = 0
for i in range(10):
    fpath = f"{extraction_dir}/ua-file-extract-results-{i}.json"
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        data = json.load(f)
    for result in data.get("results", []):
        fp = result.get("path", "")
        for cls in result.get("classes", []):
            name = cls.get("name", "")
            if not name:
                continue
            cls_id = f"class:{fp}:{name}"
            if cls_id in node_map:
                cls_node = node_map[cls_id]
                methods = [m for m in cls.get("methods", []) if m and not m.startswith("_")]
                method_str = ", ".join(methods[:6])
                if methods:
                    cls_node["summary"] = f"{name} 类 — 定义在 {fp} 中。提供方法: {method_str}。"
                else:
                    cls_node["summary"] = f"{name} 类 — 定义在 {fp} 中。"
                enhanced_classes += 1

print(f"  Enhanced {enhanced_classes} class node summaries")

# ── 6. Add configures edges for config files ──
print("\n=== 6. Adding configures edges ===")
config_nodes = [n for n in nodes if n["type"] == "config"]
config_edges_added = 0
existing_config_edges = set()
for e in edges:
    if e["type"] == "configures":
        existing_config_edges.add((e["source"], e["target"]))

for cfg in config_nodes:
    fp = cfg.get("filePath", "")
    if ".cmake" in fp or "CMakeLists" in fp:
        # CMake files configure the build
        targets = ["file:torch/__init__.py", "file:torchgen/gen.py"]
        for t in targets:
            if (cfg["id"], t) not in existing_config_edges and t in node_map:
                edges.append({
                    "source": cfg["id"],
                    "target": t,
                    "type": "configures",
                    "direction": "forward",
                    "weight": 0.6,
                })
                config_edges_added += 1

print(f"  Added {config_edges_added} configures edges")

# ── Save ──
graph["nodes"] = nodes
graph["edges"] = edges

with open(KG, "w", encoding="utf-8") as f:
    json.dump(graph, f, indent=2, ensure_ascii=False)

nodes = graph["nodes"]
edges = graph["edges"]
types = {}
for n in nodes:
    t = n.get('type', '?')
    types[t] = types.get(t, 0) + 1
edge_types = {}
for e in edges:
    t = e.get('type', '?')
    edge_types[t] = edge_types.get(t, 0) + 1

print(f"\n{'='*40}")
print(f"增强后最终统计:")
print(f"  节点: {len(nodes)}")
for t, c in sorted(types.items(), key=lambda x:-x[1]):
    print(f"    {t}: {c}")
print(f"  边: {len(edges)}")
for t, c in sorted(edge_types.items(), key=lambda x:-x[1]):
    print(f"    {t}: {c}")
