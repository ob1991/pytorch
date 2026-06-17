const fs = require('fs');
const path = require('path');

const SCAN_RESULT = '/home/bluce/pytorch/.understand-anything/intermediate/scan-result.json';
const BATCH_OUT = '/home/bluce/pytorch/.understand-anything/intermediate/batch-0.json';
const IMPORTANT_FILES_OUT = '/home/bluce/pytorch/.understand-anything/tmp/important-files.json';

const data = require(SCAN_RESULT);

// Core directories/files to include
const coreDirs = [
  "torch/__init__.py", "torch/_ops.py", "torch/_classes.py", "torch/_utils.py",
  "torch/_tensor_str.py", "torch/_compile.py", "torch/_environment.py",
  "torch/_guards.py", "torch/_sources.py", "torch/_appdirs.py",
  "torch/_custom_ops.py", "torch/_meta_registrations.py",
  "torch/_weights_only_unpickler.py", "torch/_tensor_iterator.py",
  "torch/_thread_safe_fork.py", "torch/_lobpcg.py", "torch/_lowrank.py",
  "torch/hub.py", "torch/random.py", "torch/types.py", "torch/torch_version.py",
  "torch/quasirandom.py", "torch/__future__.py", "torch/__config__.py",
  "torch/library.py", "torch/functional.py", "torch/serialization.py",
  "torch/storage.py", "torch/masked.py", "torch/return_types.py",
  "torch/_lazy.py", "torch/_numpy/__init__.py",
  "torch/_inductor/", "torch/_dynamo/",
  "torch/nn/", "torch/optim/", "torch/autograd/",
  "torch/cuda/", "torch/mps/", "torch/xpu/", "torch/accelerator/",
  "torch/distributed/", "torch/fx/", "torch/export/",
  "torch/onnx/", "torch/quantization/", "torch/jit/",
  "torch/profiler/", "torch/sparse/", "torch/linalg/", "torch/fft/",
  "torch/special/", "torch/compiler/",
  "torch/package/", "torch/monitor/", "torch/distributions/",
  "torch/backends/", "torch/signal/", "torch/overrides/",
  "torch/_functorch/", "torch/_decomp/", "torch/_prims/",
  "torch/_prims_common/", "torch/_subclasses/", "torch/_refs/",
  "torch/utils/", "torch/testing/_internal/",
  "torchgen/",
  "torch/_lazy/", "torch/amp/", "torch/_strobelight/",
  "torch/_logging/", "torch/_library/", "torch/_custom_op/",
  "torch/_higher_order_ops/",
  "torch/_awaits/",
  "torch/_C/__init__.pyi.in",
];

function isCoreFile(f) {
  const p = f.path;
  if (p.endsWith('.pyi') && !p.endsWith('__init__.pyi.in')) return false;
  
  for (const dir of coreDirs) {
    if (dir.endsWith('/')) { if (p.startsWith(dir)) return true; }
    else { if (p === dir) return true; }
  }
  
  if (/^torch\/[^/]+\.py$/.test(p)) return true;
  if (p === 'pyproject.toml' || p === 'setup.py' || p === 'CMakeLists.txt' || p === 'README.md') return true;
  if (p.startsWith("torchgen/") && p.endsWith(".py")) return true;
  
  return false;
}

const coreFiles = data.files.filter(isCoreFile);
const corePaths = new Set(coreFiles.map(f => f.path));

// Node type mapping
function getNodeType(f) {
  const ext = path.extname(f.path);
  const base = path.basename(f.path);
  if (f.fileCategory === 'config') return 'config';
  if (f.fileCategory === 'docs') return 'document';
  if (f.fileCategory === 'infra') {
    if (base === 'Dockerfile') return 'service';
    if (/\.github\/workflows/.test(f.path)) return 'pipeline';
    return 'file';
  }
  if (f.fileCategory === 'data') {
    if (ext === '.sql') return 'table';
    if (ext === '.graphql' || ext === '.proto') return 'schema';
    return 'file';
  }
  return 'file';
}

function getNodePrefix(type) {
  const map = {
    'file': 'file:', 'config': 'config:', 'document': 'document:',
    'service': 'service:', 'pipeline': 'pipeline:',
    'table': 'table:', 'schema': 'schema:'
  };
  return map[type] || 'file:';
}

// Smart summary generation based on path and filename
function generateSummary(f) {
  const p = f.path;
  const base = path.basename(p);
  const dir = path.dirname(p);

  if (base === '__init__.py') {
    const pkg = dir.replace(/\//g, '.');
    return `${pkg} 包的初始化模块，定义了包的主要导出接口。`;
  }
  if (p === 'torch/__init__.py') return 'PyTorch 主入口模块，导出核心张量操作、神经网络 API 和自动微分功能。';
  if (p === 'torch/_ops.py') return '定义 PyTorch 算子注册和调度机制的内部模块。';
  if (p === 'torch/_classes.py') return '管理 PyTorch C++ 类绑定的内部模块。';
  if (p === 'torch/_utils.py') return 'PyTorch 内部工具函数集合。';
  if (p === 'torch/nn/__init__.py') return '神经网络模块的入口，导出层、损失函数和容器类。';
  if (p === 'torch/nn/modules/__init__.py') return '神经网络层模块的集合导出。';
  if (p.startsWith('torch/nn/modules/')) return `定义 ${base.replace('.py', '')} 神经网络层。`;
  if (p === 'torch/optim/__init__.py') return '优化器模块入口，导出 SGD、Adam 等优化算法。';
  if (p.startsWith('torch/optim/') && p.endsWith('.py') && base !== '__init__.py') 
    return `实现 ${base.replace('.py', '')} 优化器算法。`;
  if (p === 'torch/autograd/__init__.py') return '自动微分模块入口，导出 grad、Function 等核心 API。';
  if (p === 'torch/cuda/__init__.py') return 'CUDA 后端支持模块，提供 GPU 内存管理和设备操作。';
  if (p === 'torch/fx/__init__.py') return 'FX 符号跟踪和变换框架入口。';
  if (p === 'torch/_dynamo/__init__.py') return 'TorchDynamo 即时编译引擎入口，负责图捕获和优化。';
  if (p === 'torch/_inductor/__init__.py') return 'TorchInductor 代码生成器入口，将计算图编译为 GPU/CPU 内核。';
  if (p === 'torch/distributed/__init__.py') return '分布式训练模块入口，支持 NCCL/Gloo/MPI 后端。';
  if (p === 'torch/jit/__init__.py') return 'TorchScript 编译栈入口，支持模型序列化和部署。';
  if (p === 'torch/onnx/__init__.py') return 'ONNX 导出/导入功能入口。';
  if (p === 'torch/export/__init__.py') return 'torch.export 模型导出功能入口。';
  if (p === 'pyproject.toml') return 'PyTorch 项目元数据和构建系统配置。';
  if (p === 'setup.py') return 'Python 包构建入口脚本，集成 CMake 构建 C++ 扩展。';
  if (p === 'CMakeLists.txt') return 'CMake 根构建配置文件，管理 C++/CUDA 编译。';
  if (p === 'README.md') return 'PyTorch 项目说明文档，包含安装指南和功能概述。';
  if (p.startsWith('torchgen/')) return `代码生成工具：${base.replace('.py', '')} 模块。`;
  if (p.startsWith('torch/distributed/')) {
    const sub = p.replace('torch/distributed/', '').replace('.py', '').replace('/', '.');
    return `分布式训练 ${sub} 模块。`;
  }
  if (p.startsWith('torch/_dynamo/') && base !== '__init__.py')
    return `TorchDynamo ${base.replace('.py', '')} 子模块。`;
  if (p.startsWith('torch/_inductor/') && base !== '__init__.py')
    return `TorchInductor ${base.replace('.py', '')} 子模块。`;
  if (p.startsWith('torch/fx/') && base !== '__init__.py')
    return `FX ${base.replace('.py', '')} 子模块。`;
  if (p.startsWith('torch/utils/'))
    return `PyTorch 工具模块：${base.replace('.py', '')}。`;
  if (p.startsWith('torch/testing/'))
    return `测试支持模块：${base.replace('.py', '')}。`;
  if (p.startsWith('torch/onnx/') && base !== '__init__.py')
    return `ONNX ${base.replace('.py', '')} 子模块。`;
  if (p.startsWith('torch/export/') && base !== '__init__.py')
    return `Export ${base.replace('.py', '')} 子模块。`;
  if (p.startsWith('torch/quantization/'))
    return `量化 ${base.replace('.py', '')} 模块。`;
  if (p.startsWith('torch/profiler/'))
    return `性能分析 ${base.replace('.py', '')} 模块。`;

  return `${base} — PyTorch ${dir.replace('/', '.')} 模块。`;
}

function generateTags(f) {
  const p = f.path;
  const base = path.basename(p);
  const tags = [];
  
  if (base === '__init__.py') tags.push('entry-point', 'barrel', 'package-init');
  if (p === 'torch/__init__.py') tags.push('entry-point', 'main-api', 'core');
  if (p.startsWith('torch/nn/')) tags.push('neural-network');
  if (p.startsWith('torch/optim/')) tags.push('optimization', 'training');
  if (p.startsWith('torch/autograd/')) tags.push('autograd', 'differentiation');
  if (p.startsWith('torch/cuda/')) tags.push('cuda', 'gpu');
  if (p.startsWith('torch/fx/')) tags.push('fx', 'graph-transformation');
  if (p.startsWith('torch/_dynamo/')) tags.push('dynamo', 'jit', 'graph-capture');
  if (p.startsWith('torch/_inductor/')) tags.push('inductor', 'codegen', 'gpu-kernel');
  if (p.startsWith('torch/distributed/')) tags.push('distributed', 'parallel-training');
  if (p.startsWith('torch/jit/')) tags.push('jit', 'torchscript', 'compilation');
  if (p.startsWith('torch/onnx/')) tags.push('onnx', 'export');
  if (p.startsWith('torch/export/')) tags.push('export', 'deployment');
  if (p.startsWith('torch/quantization/')) tags.push('quantization', 'optimization');
  if (p.startsWith('torch/profiler/')) tags.push('profiler', 'performance');
  if (p.startsWith('torchgen/')) tags.push('codegen', 'tooling');
  if (p.startsWith('torch/utils/')) tags.push('utility');
  if (p.startsWith('torch/testing/')) tags.push('testing', 'infrastructure');
  if (p.startsWith('torch/_functorch/')) tags.push('functorch', 'functional-transform');
  if (p.startsWith('torch/sparse/')) tags.push('sparse', 'tensor');
  if (p.startsWith('torch/linalg/')) tags.push('linalg', 'linear-algebra');
  if (p.startsWith('torch/_decomp/')) tags.push('decomposition');
  if (p.startsWith('torch/_prims/')) tags.push('primitive-ops');
  if (p === 'pyproject.toml' || p === 'setup.py') tags.push('build-system', 'configuration');
  if (p === 'CMakeLists.txt') tags.push('build-system', 'cmake');
  if (p === 'README.md' || p === 'CONTRIBUTING.md') tags.push('documentation');
  if (base.endsWith('.pyi') || base.endsWith('.pyi.in')) tags.push('type-stubs');
  if (p.startsWith('torch/_higher_order_ops/')) tags.push('higher-order-ops');
  
  if (tags.length === 0) tags.push('module', 'python');
  return [...new Set(tags)];
}

function determineComplexity(f) {
  if (f.sizeLines > 500) return 'complex';
  if (f.sizeLines > 100) return 'moderate';
  return 'simple';
}

// Build nodes
const nodes = [];
const edges = [];
const fileNodeIds = new Set();

for (const f of coreFiles) {
  const type = getNodeType(f);
  const prefix = getNodePrefix(type);
  const nodeId = prefix + f.path;
  
  nodes.push({
    id: nodeId,
    type: type,
    name: path.basename(f.path),
    filePath: f.path,
    summary: generateSummary(f),
    tags: generateTags(f),
    complexity: determineComplexity(f)
  });
  
  fileNodeIds.add(nodeId);
}

// Build import edges from import map
for (const f of coreFiles) {
  if (!data.importMap[f.path]) continue;
  const sourcePrefix = getNodePrefix(getNodeType(f));
  const sourceId = sourcePrefix + f.path;
  
  for (const targetPath of data.importMap[f.path]) {
    // Only create edges to files in our set
    const targetFile = coreFiles.find(cf => cf.path === targetPath);
    if (!targetFile) continue;
    
    const targetType = getNodeType(targetFile);
    const targetPrefix = getNodePrefix(targetType);
    const targetId = targetPrefix + targetPath;
    
    if (fileNodeIds.has(targetId)) {
      edges.push({
        source: sourceId,
        target: targetId,
        type: 'imports',
        direction: 'forward',
        weight: 0.7
      });
    }
  }
}

console.error(`Generated ${nodes.length} nodes and ${edges.length} edges`);

// Count by type
const typeCounts = {};
for (const n of nodes) {
  typeCounts[n.type] = (typeCounts[n.type] || 0) + 1;
}
console.error('Node types:', JSON.stringify(typeCounts));

// Write batch output
const output = { nodes, edges };
fs.writeFileSync(BATCH_OUT, JSON.stringify(output, null, 2));

// Also write list of important file paths for later phases
const importantFiles = coreFiles.map(f => ({
  path: f.path,
  language: f.language,
  sizeLines: f.sizeLines,
  fileCategory: f.fileCategory
}));
fs.writeFileSync(IMPORTANT_FILES_OUT, JSON.stringify(importantFiles, null, 2));

console.log(`Written to ${BATCH_OUT}`);
