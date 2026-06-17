#!/usr/bin/env python3
"""Print PyTorch doc reading roadmap with file paths."""
import json
from pathlib import Path

# The curated reading list with verified file paths
roadmap = [
    ("阶段 1：基础入门", [
        ("核心 API", [
            ("张量核心 API", "docs/source/torch.md"),
            ("张量详解", "docs/source/tensors.md"),
            ("张量属性", "docs/source/tensor_attributes.md"),
            ("扩展机制", "docs/source/library.md"),
        ]),
        ("自动微分", [
            ("autograd 引擎", "docs/source/autograd.md"),
            ("autograd 进阶", "docs/source/notes/autograd.md"),
        ]),
        ("CUDA GPU", [
            ("CUDA 支持", "docs/source/cuda.md"),
            ("CUDA 最佳实践", "docs/source/notes/cuda.md"),
        ]),
    ]),
    ("阶段 2：模块构建", [
        ("神经网络", [
            ("nn 模块", "docs/source/nn.md"),
            ("注意力机制", "docs/source/nn.attention.md"),
            ("FlexAttention", "docs/source/nn.attention.flex_attention.md"),
        ]),
        ("FX 图捕获", [
            ("FX 中间表示", "docs/source/fx.md"),
            ("FX 实验功能", "docs/source/fx.experimental.md"),
        ]),
    ]),
    ("阶段 3：编译与优化", [
        ("torch.compile", [
            ("编译管线总览", "docs/source/torch.compiler_api.md"),
        ]),
        ("导出与部署", [
            ("torch.export", "docs/source/user_guide/torch_compiler/export.md"),
            ("ONNX 导出", "docs/source/onnx.md"),
        ]),
        ("量化", [
            ("模型量化", "docs/source/quantization.md"),
            ("量化算子支持", "docs/source/quantization-support.md"),
        ]),
    ]),
    ("阶段 4：分布式与高级特性", [
        ("分布式训练", [
            ("分布式基础", "docs/source/distributed.md"),
            ("DTensor", "docs/source/distributed.tensor.md"),
            ("Pipeline 并行", "docs/source/distributed.pipelining.md"),
            ("弹性训练", "docs/source/distributed.elastic.md"),
        ]),
        ("函数式变换", [
            ("vmap/grad API", "docs/source/func.api.md"),
            ("使用限制", "docs/source/func.ux_limitations.md"),
        ]),
        ("特殊操作", [
            ("稀疏张量", "docs/source/sparse.md"),
            ("概率分布", "docs/source/distributions.md"),
            ("线性代数", "docs/source/linalg.md"),
        ]),
    ]),
    ("阶段 5：工具与实践", [
        ("工具层", [
            ("DataLoader", "docs/source/data.md"),
            ("性能分析器", "docs/source/profiler.md"),
            ("梯度检查点", "docs/source/checkpoint.md"),
            ("TensorBoard", "docs/source/tensorboard.md"),
        ]),
    ]),
]

# Verify paths exist
proj_root = Path("/home/bluce/pytorch")

print("=" * 58)
print("📖  PyTorch 文档精读路线（附文件路径）")
print("=" * 58)

num = 0
for phase_name, layers in roadmap:
    print(f"\n{'─' * 58}")
    print(f"【{phase_name}】")
    print(f"{'─' * 58}")
    for layer_name, docs in layers:
        print(f"\n  ▸ {layer_name}")
        for title, path in docs:
            num += 1
            full_path = f"docs/{path}" if not path.startswith("docs/") else path
            exists = (proj_root / path).exists()
            marker = "" if exists else " ⚠️"
            print(f"    {num:2d}. [{title}]({path}){marker}")

print(f"\n{'=' * 58}")
print(f"共 {num} 篇文档，可在 IDE 中点击路径直接打开")
print(f"{'=' * 58}")
