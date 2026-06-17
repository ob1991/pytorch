#!/usr/bin/env python3
"""
Build layers.json directly from directory groups and PyTorch knowledge.
Extracts file IDs from the JSON input file using simple grep-style operations.
"""
import json
import os
import subprocess
import sys

input_path = '/home/bluce/pytorch/.understand-anything/tmp/arch-input.json'
output_path = '/home/bluce/pytorch/.understand-anything/intermediate/layers.json'

# Read all file IDs from the input
import re

with open(input_path, 'r') as f:
    content = f.read()

# Extract all node IDs
node_ids = re.findall(r'"id":\s*"([^"]+)"', content)
print(f"Total node IDs extracted: {len(node_ids)}", file=sys.stderr)

# Categorize by type
file_ids = [nid for nid in node_ids if nid.startswith('file:')]
config_ids = [nid for nid in node_ids if nid.startswith('config:')]
document_ids = [nid for nid in node_ids if nid.startswith('document:')]

print(f"File IDs: {len(file_ids)}", file=sys.stderr)
print(f"Config IDs: {len(config_ids)}", file=sys.stderr)
print(f"Document IDs: {len(document_ids)}", file=sys.stderr)

# Group file IDs by directory
def get_dir_group(nid):
    path = nid.split(':', 1)[1]
    if path.startswith('torch/'):
        rest = path[6:]  # Remove 'torch/'
        if '/' in rest:
            return rest.split('/')[0]
        else:
            return '__root__'
    elif path.startswith('torchgen/'):
        rest = path[9:]  # Remove 'torchgen/'
        if '/' in rest:
            return rest.split('/')[0]
        else:
            return '__torchgen_root__'
    else:
        return '__other__'

dir_groups = {}
for nid in file_ids:
    group = get_dir_group(nid)
    dir_groups.setdefault(group, []).append(nid)

print(f"\nDirectory groups: {len(dir_groups)}", file=sys.stderr)
for g in sorted(dir_groups.keys(), key=lambda x: -len(dir_groups[x])):
    print(f"  {g}: {len(dir_groups[g])}", file=sys.stderr)

# Also group document and config IDs
doc_groups = {}
for nid in document_ids:
    group = get_dir_group(nid)
    doc_groups.setdefault(group, []).append(nid)
for nid in config_ids:
    group = get_dir_group(nid)
    doc_groups.setdefault(group, []).append(nid)

# BUILD LAYERS
layers = []

# 1. Core Layer (核心层)
# Includes: __root__ files, _C, _awaits, autograd, _subclasses, _custom_op,
# _library, _higher_order_ops, _refs, _prims, _prims_common,
# sparse, linalg, fft, signal, special, _numpy, _decomp (core decomps),
# overrides.py, library.py, types.py, _ops.py, storage.py, _tensor.py, etc.
core_dirs = ['__root__', '_C', '_awaits', 'autograd', '_subclasses',
             '_custom_op', '_library', '_higher_order_ops',
             '_refs', '_prims', '_prims_common',
             'sparse', 'linalg', 'fft', 'signal', 'special',
             '_numpy']
core_ids = []
for d in core_dirs:
    if d in dir_groups:
        core_ids.extend(dir_groups[d])

layers.append({
    "id": "layer:core",
    "name": "核心层",
    "description": "PyTorch 核心张量操作、自动微分、算子系统、存储管理和数据类型定义",
    "nodeIds": sorted(core_ids)
})

# 2. Neural Network Layer (神经网络层)
nn_dirs = ['nn', 'optim', 'distributions']
nn_ids = []
for d in nn_dirs:
    if d in dir_groups:
        nn_ids.extend(dir_groups[d])
layers.append({
    "id": "layer:neural-network",
    "name": "神经网络层",
    "description": "神经网络模块、优化器和概率分布的实现",
    "nodeIds": sorted(nn_ids)
})

# 3. Compilation Layer (编译栈层)
comp_dirs = ['_dynamo', '_inductor', 'fx', '_functorch', '_decomp', 'compiler']
comp_ids = []
for d in comp_dirs:
    if d in dir_groups:
        comp_ids.extend(dir_groups[d])
# Add documents and config for these dirs
for d in comp_dirs:
    if d in doc_groups:
        comp_ids.extend(doc_groups[d])
layers.append({
    "id": "layer:compilation",
    "name": "编译栈层",
    "description": "TorchDynamo JIT 图捕获、TorchInductor 代码生成、FX 图变换和函数变换 (functorch)",
    "nodeIds": sorted(comp_ids)
})

# 4. Distributed Layer (分布式层)
dist_ids = list(dir_groups.get('distributed', []))
if 'distributed' in doc_groups:
    dist_ids.extend(doc_groups['distributed'])
layers.append({
    "id": "layer:distributed",
    "name": "分布式层",
    "description": "分布式训练、集合通信、FSDP、RPC、流水线并行和张量并行",
    "nodeIds": sorted(dist_ids)
})

# 5. GPU and Backend Layer (GPU与后端层)
gpu_dirs = ['cuda', 'xpu', 'mps', 'backends', 'accelerator']
gpu_ids = []
for d in gpu_dirs:
    if d in dir_groups:
        gpu_ids.extend(dir_groups[d])
if 'cuda' in doc_groups:
    gpu_ids.extend(doc_groups['cuda'])
layers.append({
    "id": "layer:gpu-backend",
    "name": "GPU与后端层",
    "description": "CUDA、XPU、MPS 等硬件后端的支持和设备管理",
    "nodeIds": sorted(gpu_ids)
})

# 6. Export and Deployment Layer (导出与部署层)
export_dirs = ['onnx', 'export', 'jit']
export_ids = []
for d in export_dirs:
    if d in dir_groups:
        export_ids.extend(dir_groups[d])
for d in export_dirs:
    if d in doc_groups:
        doc_ids_for_export = [nid for nid in doc_groups[d] if nid.startswith('document:')]
        export_ids.extend(doc_ids_for_export)
layers.append({
    "id": "layer:export",
    "name": "导出与部署层",
    "description": "ONNX 导出、TorchScript、模型导出和序列化",
    "nodeIds": sorted(export_ids)
})

# 7. Quantization and Optimization Layer (量化与优化层)
quant_dirs = ['quantization', 'amp']
quant_ids = []
for d in quant_dirs:
    if d in dir_groups:
        quant_ids.extend(dir_groups[d])
layers.append({
    "id": "layer:optimization",
    "name": "量化与优化层",
    "description": "模型量化、自动混合精度训练和性能优化",
    "nodeIds": sorted(quant_ids)
})

# 8. Tooling and Testing Layer (工具与测试层)
# Remaining documents that haven't been assigned yet
tool_dirs = ['utils', 'testing', 'profiler', 'package', '_logging', '_strobelight', 'monitor']
tool_ids = []
for d in tool_dirs:
    if d in dir_groups:
        tool_ids.extend(dir_groups[d])
# Add remaining documents for these dirs
for d in tool_dirs:
    if d in doc_groups:
        doc_ids_for_tool = [nid for nid in doc_groups[d] if nid.startswith('document:')]
        tool_ids.extend(doc_ids_for_tool)
layers.append({
    "id": "layer:tooling",
    "name": "工具与测试层",
    "description": "性能分析器、测试框架、打包工具、日志记录和实用工具",
    "nodeIds": sorted(tool_ids)
})

# 9. Code Generation Layer (代码生成层)
codegen_dirs = ['__torchgen_root__', '_autoheuristic', 'api', 'dest', 'static_runtime',
                'selective_build', 'operator_versions', 'aoti', 'shape_functions',
                'fuse', 'decompositions', '_lazy']
codegen_ids = []
# Get torchgen root files
if '__torchgen_root__' in dir_groups:
    codegen_ids.extend(dir_groups['__torchgen_root__'])
# Get torchgen subdirectories from doc_groups (which also has file IDs via their own dir)
torchgen_file_ids = [nid for nid in file_ids if nid.startswith('file:torchgen/')]
for nid in torchgen_file_ids:
    codegen_ids.append(nid)
# Add torchgen documents
torchgen_doc_ids = [nid for nid in document_ids if 'torchgen' in nid]
codegen_ids.extend(torchgen_doc_ids)
# Add _lazy files
if '_lazy' in dir_groups:
    codegen_ids.extend(dir_groups['_lazy'])

layers.append({
    "id": "layer:codegen",
    "name": "代码生成层",
    "description": "原生函数代码生成工具 (torchgen)、延迟后端和自动调优",
    "nodeIds": sorted(set(codegen_ids))
})

# Verify: count all assigned IDs
all_assigned = set()
for layer in layers:
    for nid in layer['nodeIds']:
        all_assigned.add(nid)

print(f"\nTotal assigned: {len(all_assigned)}", file=sys.stderr)
print(f"Total expected: {len(node_ids)}", file=sys.stderr)

if len(all_assigned) == len(node_ids):
    print("All files assigned correctly!", file=sys.stderr)
else:
    missing = set(node_ids) - all_assigned
    extra = all_assigned - set(node_ids)
    print(f"WARNING: Missing {len(missing)} files", file=sys.stderr)
    print(f"WARNING: Extra {len(extra)} files", file=sys.stderr)
    if missing:
        print(f"Missing files:", file=sys.stderr)
        for m in sorted(missing)[:20]:
            print(f"  {m}", file=sys.stderr)
    if extra:
        print(f"Extra files:", file=sys.stderr)
        for e in sorted(extra)[:20]:
            print(f"  {e}", file=sys.stderr)

# Write output
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(layers, f, ensure_ascii=False, indent=2)

print(f"\nLayers written to {output_path}", file=sys.stderr)
print(f"Number of layers: {len(layers)}", file=sys.stderr)
for layer in layers:
    print(f"  {layer['id']}: {len(layer['nodeIds'])} files", file=sys.stderr)
