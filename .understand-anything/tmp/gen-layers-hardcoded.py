#!/usr/bin/env python3
"""
Generate layers.json with hardcoded file-to-layer mappings.
Uses subprocess to extract file IDs from arch-input.json.
"""
import json
import os
import subprocess
import sys

input_path = '/home/bluce/pytorch/.understand-anything/tmp/arch-input.json'
output_path = '/home/bluce/pytorch/.understand-anything/intermediate/layers.json'

# Extract all node IDs via subprocess grep
try:
    proc = subprocess.run(
        ['grep', '-oP', r'"id":\s*"\K[^"]+', input_path],
        capture_output=True, text=True, timeout=120
    )
except Exception as e:
    print(f"Error running grep: {e}", file=sys.stderr)
    sys.exit(1)

if proc.returncode != 0:
    print(f"grep failed: {proc.stderr}", file=sys.stderr)
    sys.exit(1)

all_ids = [l.strip() for l in proc.stdout.strip().split('\n') if l.strip()]
print(f"Total: {len(all_ids)}", file=sys.stderr)

# For each ID, determine its directory group
def get_group(nid):
    path = nid.split(':', 1)[1]
    if path.startswith('torch/'):
        rest = path[6:]
        if '/' in rest:
            return rest.split('/')[0]
        return '__root__'
    elif path.startswith('torchgen/'):
        return '__torchgen__'
    return None

# Group all IDs
grouped = {}
for nid in all_ids:
    g = get_group(nid)
    if g:
        grouped.setdefault(g, []).append(nid)

# Print summary
for g in sorted(grouped, key=lambda x: -len(grouped[x])):
    print(f"  {g}: {len(grouped[g])}", file=sys.stderr)

# Define layer structure
layer_defs = [
    ("layer:core", "核心层",
     "PyTorch 核心张量操作、自动微分、算子系统、存储管理和数据类型定义",
     ['__root__', 'autograd', '_subclasses', '_custom_op', '_library',
      '_higher_order_ops', '_refs', '_prims', '_prims_common',
      'sparse', 'linalg', 'fft', 'signal', 'special',
      '_numpy', '_C', '_awaits', 'monitor']),
    ("layer:neural-network", "神经网络层",
     "神经网络模块、优化算法和概率分布的定义与实现",
     ['nn', 'optim', 'distributions']),
    ("layer:compilation", "编译栈层",
     "TorchDynamo JIT 图捕获、TorchInductor 代码生成、FX 图变换和函数变换 (functorch)",
     ['_dynamo', '_inductor', 'fx', '_functorch', '_decomp', 'compiler']),
    ("layer:distributed", "分布式层",
     "分布式训练、集合通信、FSDP、RPC、流水线并行和张量并行",
     ['distributed']),
    ("layer:gpu-backend", "GPU与后端层",
     "CUDA、XPU、MPS 等硬件后端的支持、内存管理和设备抽象",
     ['cuda', 'xpu', 'mps', 'backends', 'accelerator']),
    ("layer:export", "导出与部署层",
     "ONNX 导出、TorchScript 编译、模型导出和序列化格式转换",
     ['onnx', 'export', 'jit']),
    ("layer:optimization", "量化与优化层",
     "模型量化、自动混合精度训练和运行时性能优化",
     ['quantization', 'amp']),
    ("layer:tooling", "工具与测试层",
     "测试工具、性能分析器、数据加载、打包分发和日志记录基础设施",
     ['utils', 'testing', 'profiler', 'package', '_logging', '_strobelight']),
    ("layer:codegen", "代码生成层",
     "原生函数代码生成工具 (torchgen)、延迟后端编译和自动调优启发式",
     ['__torchgen__', '_lazy']),
]

layers = []
all_assigned = set()

for lid, lname, ldesc, dirs in layer_defs:
    node_ids = []
    for d in dirs:
        if d in grouped:
            node_ids.extend(grouped[d])
    node_ids = sorted(set(node_ids))
    for nid in node_ids:
        all_assigned.add(nid)
    layers.append({
        "id": lid,
        "name": lname,
        "description": ldesc,
        "nodeIds": node_ids
    })

# Check for missing
missing = set(all_ids) - all_assigned
if missing:
    print(f"\nWARNING: {len(missing)} unassigned IDs:", file=sys.stderr)
    for m in sorted(missing):
        print(f"  {m}", file=sys.stderr)

print(f"\nAssigned: {len(all_assigned)} / {len(all_ids)}", file=sys.stderr)
print(f"Layers: {len(layers)}", file=sys.stderr)
for l in layers:
    print(f"  {l['id']}: {len(l['nodeIds'])}", file=sys.stderr)

if len(all_assigned) != len(all_ids):
    print("ERROR: Count mismatch!", file=sys.stderr)
    sys.exit(1)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(layers, f, ensure_ascii=False, indent=2)
print(f"Written to {output_path}", file=sys.stderr)
sys.exit(0)
