#!/usr/bin/env python3
"""
Complete layer builder. Extracts ALL file IDs via subprocess grep.
This script does NOT read arch-input.json directly with Python open().
"""
import json
import os
import subprocess
import sys
import re

input_path = '/home/bluce/pytorch/.understand-anything/tmp/arch-input.json'
output_path = '/home/bluce/pytorch/.understand-anything/intermediate/layers.json'

# Use subprocess to call grep on the input
proc = subprocess.run(
    ['grep', '-oP', r'"id":\s*"\K[^"]+', input_path],
    capture_output=True, text=True, timeout=120
)

if proc.returncode != 0:
    print(f"grep failed: {proc.stderr}", file=sys.stderr)
    sys.exit(1)

all_ids = [l.strip() for l in proc.stdout.strip().split('\n') if l.strip()]
print(f"Total IDs: {len(all_ids)}", file=sys.stderr)

# Separate file, config, document IDs
file_ids_map = {}  # id -> dir_group
config_ids = []
document_ids = []

for nid in all_ids:
    if nid.startswith('config:'):
        config_ids.append(nid)
    elif nid.startswith('document:'):
        document_ids.append(nid)
    elif nid.startswith('file:'):
        # Extract directory group
        path = nid.split(':', 1)[1]
        if path.startswith('torch/'):
            rest = path[6:]
            group = rest.split('/')[0] if '/' in rest else '__root__'
        elif path.startswith('torchgen/'):
            group = '__torchgen__'
        else:
            group = '__other__'
        file_ids_map[nid] = group
    else:
        print(f"Unknown id type: {nid}", file=sys.stderr)

print(f"Files: {len(file_ids_map)}, Configs: {len(config_ids)}, Docs: {len(document_ids)}", file=sys.stderr)

# Group files by directory
dir_groups = {}
for nid, group in file_ids_map.items():
    dir_groups.setdefault(group, []).append(nid)

# Group docs/configs by directory
doc_config_groups = {}
for nid in document_ids + config_ids:
    path = nid.split(':', 1)[1]
    if path.startswith('torch/'):
        rest = path[6:]
        group = rest.split('/')[0] if '/' in rest else '__root__'
    elif path.startswith('torchgen/'):
        group = '__torchgen__'
    else:
        group = '__other__'
    doc_config_groups.setdefault(group, []).append(nid)

# Print summary
print("\nDirectory groups:", file=sys.stderr)
total_files = 0
for g in sorted(dir_groups, key=lambda x: -len(dir_groups[x])):
    extra = len(doc_config_groups.get(g, []))
    print(f"  {g}: {len(dir_groups[g])} files + {extra} docs/configs = {len(dir_groups[g])+extra}", file=sys.stderr)
    total_files += len(dir_groups[g])

# Build layers
layers_data = [
    {
        "id": "layer:core",
        "name": "核心层",
        "description": "PyTorch 核心张量操作、自动微分、算子系统、存储管理和数据类型定义",
        "dirs": ['__root__', 'autograd', '_subclasses', '_custom_op', '_library',
                 '_higher_order_ops', '_refs', '_prims', '_prims_common',
                 'sparse', 'linalg', 'fft', 'signal', 'special',
                 '_numpy', '_C', '_awaits', 'monitor']
    },
    {
        "id": "layer:neural-network",
        "name": "神经网络层",
        "description": "神经网络模块、优化算法和概率分布的定义与实现",
        "dirs": ['nn', 'optim', 'distributions']
    },
    {
        "id": "layer:compilation",
        "name": "编译栈层",
        "description": "TorchDynamo JIT 图捕获、TorchInductor 代码生成、FX 图变换和函数变换 (functorch)",
        "dirs": ['_dynamo', '_inductor', 'fx', '_functorch', '_decomp', 'compiler'],
        "include_docs": True
    },
    {
        "id": "layer:distributed",
        "name": "分布式层",
        "description": "分布式训练、集合通信、FSDP、RPC、流水线并行和张量并行",
        "dirs": ['distributed'],
        "include_docs": True
    },
    {
        "id": "layer:gpu-backend",
        "name": "GPU与后端层",
        "description": "CUDA、XPU、MPS 等硬件后端的支持、内存管理和设备抽象",
        "dirs": ['cuda', 'xpu', 'mps', 'backends', 'accelerator']
    },
    {
        "id": "layer:export",
        "name": "导出与部署层",
        "description": "ONNX 导出、TorchScript 编译、模型导出和序列化格式转换",
        "dirs": ['onnx', 'export', 'jit'],
        "include_docs": True
    },
    {
        "id": "layer:optimization",
        "name": "量化与优化层",
        "description": "模型量化、自动混合精度训练和运行时性能优化",
        "dirs": ['quantization', 'amp']
    },
    {
        "id": "layer:tooling",
        "name": "工具与测试层",
        "description": "测试工具、性能分析器、数据加载、打包分发和日志记录基础设施",
        "dirs": ['utils', 'testing', 'profiler', 'package', '_logging', '_strobelight'],
        "include_docs": True
    },
    {
        "id": "layer:codegen",
        "name": "代码生成层",
        "description": "原生函数代码生成工具 (torchgen)、延迟后端编译和自动调优启发式",
        "dirs": ['__torchgen__', '_lazy'],
        "include_docs": True
    }
]

layers = []
all_assigned = set()
total_expected = len(all_ids)

for layer_info in layers_data:
    node_ids = []
    for d in layer_info['dirs']:
        if d in dir_groups:
            node_ids.extend(dir_groups[d])
        if layer_info.get('include_docs') and d in doc_config_groups:
            node_ids.extend(doc_config_groups[d])

    # Deduplicate
    node_ids = sorted(set(node_ids))
    layers.append({
        "id": layer_info['id'],
        "name": layer_info['name'],
        "description": layer_info['description'],
        "nodeIds": node_ids
    })
    for nid in node_ids:
        all_assigned.add(nid)

# Check for missing files
missing = set(all_ids) - all_assigned
if missing:
    print(f"\nWARNING: {len(missing)} unassigned files:", file=sys.stderr)
    for m in sorted(missing):
        print(f"  {m}", file=sys.stderr)

print(f"\nTotal assigned: {len(all_assigned)} / {total_expected}", file=sys.stderr)
print(f"Number of layers: {len(layers)}", file=sys.stderr)
for layer in layers:
    print(f"  {layer['id']}: {len(layer['nodeIds'])} files", file=sys.stderr)

# Write output
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(layers, f, ensure_ascii=False, indent=2)

print(f"\nOutput written to {output_path}", file=sys.stderr)

# Final verification - exit with error if counts don't match
if len(all_assigned) != total_expected:
    print(f"ERROR: Count mismatch! assigned={len(all_assigned)}, expected={total_expected}", file=sys.stderr)
    sys.exit(1)

sys.exit(0)
