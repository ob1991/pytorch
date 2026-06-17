#!/usr/bin/env python3
"""
Deep enhancement: improve function/class summaries for key nodes,
add more semantic edges, and enrich tags.
"""
import json, re, os, textwrap

KG = "/home/bluce/pytorch/.understand-anything/knowledge-graph.json"
PROJECT = "/home/bluce/pytorch"

with open(KG) as f:
    graph = json.load(f)

nodes = graph["nodes"]
edges = graph["edges"]
node_map = {n["id"]: n for n in nodes}

# ── Known function descriptions (from PyTorch source knowledge) ──
FUNCTION_DESCRIPTIONS = {
    # torch/__init__.py
    "torch/__init__.py:sym_not": "符号布尔取反。在符号形状推理中用于逻辑非运算。",
    "torch/__init__.py:sym_float": "将值转换为符号浮点数。在动态形状推理中用于类型提升。",
    "torch/__init__.py:sym_int": "将值转换为符号整数。用于动态形状的整数运算。",
    "torch/__init__.py:sym_max": "符号最大值计算。在动态形状中计算张量维度的最大值。",
    "torch/__init__.py:sym_min": "符号最小值计算。在动态形状中计算张量维度的最小值。",

    # torch/_tensor.py
    "torch/_tensor.py:Tensor__add__": "实现张量加法运算（+ 操作符）。支持广播和类型提升。",
    "torch/_tensor.py:Tensor__mul__": "实现张量乘法运算（* 操作符）。支持广播。",
    "torch/_tensor.py:Tensor__matmul__": "实现张量矩阵乘法（@ 操作符）。",
    "torch/_tensor.py:Tensor__getitem__": "实现张量索引操作（[]）。支持整数、切片、布尔掩码和张量索引。",
    "torch/_tensor.py:Tensor__setitem__": "实现张量赋值操作（[] = 值）。",
    "torch/_tensor.py:Tensor__neg__": "实现张量取负操作（- 操作符）。",
    "torch/_tensor.py:Tensor__repr__": "张量的字符串表示。用于交互式显示。",
    "torch/_tensor.py:Tensor.size": "返回张量的形状。支持 dim 参数获取特定维度大小。",
    "torch/_tensor.py:Tensor.shape": "属性，返回描述张量各维度大小的 torch.Size 对象。",
    "torch/_tensor.py:Tensor.dtype": "属性，返回张量的数据类型（如 torch.float32）。",
    "torch/_tensor.py:Tensor.device": "属性，返回张量所在的设备（CPU/GPU）。",
    "torch/_tensor.py:Tensor.to": "张量类型/设备转换。将张量转换为指定的 dtype、device 或 layout。",
    "torch/_tensor.py:Tensor.cuda": "将张量移动到 CUDA 设备。支持指定 GPU 设备号。",
    "torch/_tensor.py:Tensor.cpu": "将张量移动到 CPU。",
    "torch/_tensor.py:Tensor.numpy": "将张量转换为 NumPy 数组。仅适用于 CPU 张量。",
    "torch/_tensor.py:Tensor.item": "将单元素张量转换为 Python 标量。",
    "torch/_tensor.py:Tensor.view": "重塑张量而不改变数据。返回共享数据存储的新视图。",
    "torch/_tensor.py:Tensor.reshape": "重塑张量。返回视图（可能时）或副本。",
    "torch/_tensor.py:Tensor.contiguous": "确保张量在内存中是连续的。",
    "torch/_tensor.py:Tensor.clone": "复制张量。返回具有相同数据的新张量。",
    "torch/_tensor.py:Tensor.detach": "从计算图中分离。返回不追踪梯度的新张量。",
    "torch/_tensor.py:Tensor.requires_grad_": "就地修改张量的梯度追踪设置。",
    "torch/_tensor.py:Tensor.backward": "启动反向传播。计算张量的梯度。",
    "torch/_tensor.py:Tensor.grad": "属性，返回梯度张量。",
    "torch/_tensor.py:Tensor.__deepcopy__": "支持深拷贝张量操作。",
    "torch/_tensor.py:Tensor.__reduce__": "张量的序列化协议。用于 pickle 序列化。",
    "torch/_tensor.py:Tensor.is_cuda": "属性，检查张量是否在 CUDA 设备上。",
    "torch/_tensor.py:Tensor.is_quantized": "属性，检查张量是否为量化张量。",
    "torch/_tensor.py:Tensor.is_sparse": "属性，检查张量是否为稀疏张量。",
    "torch/_tensor.py:Tensor.element_size": "返回单个元素的字节大小。",
    "torch/_tensor.py:Tensor.numel": "返回张量中元素总数。",
    "torch/_tensor.py:Tensor.ndimension": "返回张量的维度数。与 dim() 相同。",
    "torch/_tensor.py:Tensor.dim": "返回张量的维度数。",
    "torch/_tensor.py:Tensor.nelement": "返回张量元素数量，与 numel 相同。",
    "torch/_tensor.py:Tensor.type": "将张量转换为指定类型。支持 torch.FloatTensor 等旧式类型转换。",
    "torch/_tensor.py:Tensor.float": "将张量转换为 32 位浮点类型。",
    "torch/_tensor.py:Tensor.double": "将张量转换为 64 位浮点类型。",
    "torch/_tensor.py:Tensor.long": "将张量转换为 64 位整数类型。",
    "torch/_tensor.py:Tensor.int": "将张量转换为 32 位整数类型。",
    "torch/_tensor.py:Tensor.bool": "将张量转换为布尔类型。",
    "torch/_tensor.py:Tensor.half": "将张量转换为 16 位浮点（半精度）类型。",
    "torch/_tensor.py:Tensor.bfloat16": "将张量转换为 BFloat16 类型。",
    "torch/_tensor.py:Tensor.short": "将张量转换为 16 位整数类型。",
    "torch/_tensor.py:Tensor.char": "将张量转换为 8 位整数（字符）类型。",
    "torch/_tensor.py:Tensor.byte": "将张量转换为 8 位无符号整数类型。",
    "torch/_tensor.py:Tensor._make_subclass": "创建 Tensor 子类的内部方法。",
    "torch/_tensor.py:Tensor.__format__": "支持 format() 内置函数的张量格式化。",
    "torch/_tensor.py:Tensor.__dlpack__": "支持 DLPack 协议，用于跨框架张量交换。",
    "torch/_tensor.py:Tensor.__dlpack_device__": "返回 DLPack 协议中的设备信息。",
    "torch/_tensor.py:Tensor.mH": "返回厄米共轭转置（矩阵的共轭转置）。",
    "torch/_tensor.py:Tensor.mT": "返回矩阵的最后两维的转置。",
    "torch/_tensor.py:Tensor.H": "返回 2D 矩阵的厄米共轭转置。",
    "torch/_tensor.py:Tensor.matrixtranspose": "矩阵转置。返回最后两维交换的视图。",

    # torch/_ops.py
    "torch/_ops.py:_OpGraph": "算子图结构。管理操作符的依赖关系图。",
    "torch/_ops.py:_OpNamespace": "算子命名空间。组织和分组相关操作符。",
    "torch/_ops.py:OpOverload": "操作符重载。表示一个操作符的特定重载版本（由模式签名区分）。",
    "torch/_ops.py:OpOverloadPacket": "操作符重载包。包含一个操作符名的所有重载变体。",
    "torch/_ops.py:HigherOrderOperator": "高阶操作符。接受函数作为参数的操作符（如 vmap、grad）。",

    # torch/cuda/
    "torch/cuda/__init__.py:is_available": "检查 CUDA 是否可用。返回 True 表示系统支持 CUDA。",
    "torch/cuda/__init__.py:device_count": "返回可用的 CUDA GPU 数量。",
    "torch/cuda/__init__.py:current_device": "返回当前选定的 CUDA 设备索引。",
    "torch/cuda/__init__.py:synchronize": "同步所有 CUDA 设备。等待所有设备上的操作完成。",
    "torch/cuda/__init__.py:set_device": "设置当前 CUDA 设备。后续操作将在该设备上执行。",
    "torch/cuda/__init__.py:empty_cache": "清空 CUDA 缓存分配器中的所有未使用缓存。释放可用的 GPU 内存。",
    "torch/cuda/__init__.py:memory_allocated": "返回当前已分配的 GPU 内存量（字节）。",
    "torch/cuda/__init__.py:memory_reserved": "返回 CUDA 缓存分配器预留的 GPU 内存量（字节）。",
    "torch/cuda/__init__.py:max_memory_allocated": "返回自程序开始以来最大 GPU 内存分配量。",
    "torch/cuda/__init__.py:reset_peak_memory_stats": "重置 CUDA 内存峰值统计。",
    "torch/cuda/__init__.py:get_device_capability": "获取 CUDA 设备的计算能力（如 (8,0) 表示 Ampere）。",
    "torch/cuda/__init__.py:get_device_name": "获取 CUDA 设备名称（如 'NVIDIA A100'）。",
    "torch/cuda/__init__.py:get_device_properties": "获取 CUDA 设备属性（包括计算能力、内存大小等）。",
    "torch/cuda/__init__.py:Stream": "CUDA 流。支持异步并发执行。",
    "torch/cuda/__init__.py:Event": "CUDA 事件。用于计时和同步。",
    "torch/cuda/__init__.py:current_stream": "返回当前 CUDA 流。",
    "torch/cuda/__init__.py:default_stream": "返回默认 CUDA 流。",
    "torch/cuda/__init__.py:set_stream": "设置当前 CUDA 流。",

    # torch/autograd/
    "torch/autograd/__init__.py:backward": "计算张量的梯度。启动反向传播过程。",
    "torch/autograd/__init__.py:grad": "计算并返回梯度。不修改张量的 grad 属性。",
    "torch/autograd/function.py:Function": "自定义 autograd Function 的基类。通过定义前向和反向传播来创建自定义微分函数。",
    "torch/autograd/grad_mode.py:no_grad": "禁用梯度计算的上下文管理器。用于推理阶段或不需要梯度的操作。",
    "torch/autograd/grad_mode.py:enable_grad": "启用梯度计算的上下文管理器。在 no_grad 环境中重新启用梯度。",
    "torch/autograd/grad_mode.py:set_grad_enabled": "设置梯度计算是否启用的上下文管理器。",
    "torch/autograd/grad_mode.py:inference_mode": "推理模式。比 no_grad 更严格的推断模式，不使用 autograd 跟踪。",
    "torch/autograd/grad_mode.py:is_grad_enabled": "检查当前是否启用了梯度计算。",

    # torch/nn/
    "torch/nn/modules/linear.py:Linear": "线性层（全连接层）。执行权重矩阵乘法和偏置加法。",
    "torch/nn/modules/conv.py:Conv2d": "二维卷积层。用于图像和特征图的卷积操作。",
    "torch/nn/modules/conv.py:Conv1d": "一维卷积层。用于序列数据的卷积操作。",
    "torch/nn/modules/rnn.py:LSTM": "长短期记忆网络层。用于序列建模。",
    "torch/nn/modules/rnn.py:GRU": "门控循环单元层。LSTM 的简化变体。",
    "torch/nn/modules/transformer.py:Transformer": "Transformer 模型。基于自注意力的序列到序列架构。",
    "torch/nn/modules/transformer.py:TransformerEncoder": "Transformer 编码器。由多个 TransformerEncoderLayer 堆叠。",
    "torch/nn/modules/activation.py:ReLU": "修正线性单元激活函数。max(0, x)。",
    "torch/nn/modules/activation.py:GELU": "高斯误差线性单元激活函数。Transformer 中常用的激活函数。",
    "torch/nn/modules/activation.py:Dropout": "Dropout 正则化层。训练时随机将输入置零。",
    "torch/nn/modules/normalization.py:LayerNorm": "层归一化。在特征维度上进行归一化。",
    "torch/nn/modules/normalization.py:BatchNorm1d": "批归一化一维版本。用于全连接层或卷积层之后。",
    "torch/nn/modules/normalization.py:BatchNorm2d": "批归一化二维版本。用于卷积层之后。",
    "torch/nn/modules/pooling.py:MaxPool2d": "二维最大池化层。在空间维度上取最大值。",
    "torch/nn/modules/pooling.py:AvgPool2d": "二维平均池化层。在空间维度上取平均值。",
    "torch/nn/modules/loss.py:CrossEntropyLoss": "交叉熵损失。用于多分类任务。",
    "torch/nn/modules/loss.py:MSELoss": "均方误差损失。用于回归任务。",
    "torch/nn/modules/loss.py:L1Loss": "L1 损失（平均绝对误差）。用于回归任务。",
    "torch/nn/modules/container.py:Module": "所有神经网络模块的基类。提供参数管理、设备移动和状态转换接口。",
    "torch/nn/modules/container.py:Sequential": "顺序容器。按顺序执行包含的子模块。",
    "torch/nn/modules/container.py:ModuleList": "模块列表。存储子模块的有序列表。",
    "torch/nn/modules/container.py:ModuleDict": "模块字典。存储子模块的键值对。",
    "torch/nn/modules/container.py:ParameterList": "参数列表。存储可学习参数的有序列表。",
    "torch/nn/modules/container.py:ParameterDict": "参数字典。存储可学习参数的键值对。",
    "torch/nn/init.py:uniform_": "均匀分布初始化。在给定范围内均匀初始化张量。",
    "torch/nn/init.py:xavier_uniform_": "Xavier 均匀初始化。根据输入/输出维度缩放初始化。",
    "torch/nn/init.py:kaiming_uniform_": "Kaiming 均匀初始化。ReLU 激活函数推荐。",
    "torch/nn/init.py:normal_": "正态分布初始化。用给定均值和标准差的正态分布初始化。",
    "torch/nn/init.py:zeros_": "零初始化。将所有值设置为零。",
    "torch/nn/init.py:ones_": "单位初始化。将所有值设置为一。",
    "torch/nn/functional.py:relu": "ReLU 激活函数。torch.nn.functional 接口。",
    "torch/nn/functional.py:softmax": "Softmax 函数。将输入归一化为概率分布。",
    "torch/nn/functional.py:cross_entropy": "交叉熵损失函数。torch.nn.functional 接口。",
    "torch/nn/functional.py:mse_loss": "均方误差损失函数。torch.nn.functional 接口。",
    "torch/nn/functional.py:conv2d": "二维卷积操作。torch.nn.functional 接口。",
    "torch/nn/functional.py:max_pool2d": "二维最大池化操作。torch.nn.functional 接口。",
    "torch/nn/functional.py:batch_norm": "批归一化操作。torch.nn.functional 接口。",
    "torch/nn/functional.py:dropout": "Dropout 操作。torch.nn.functional 接口。",
    "torch/nn/functional.py:linear": "线性变换操作。torch.nn.functional 接口。",
    "torch/nn/functional.py:layer_norm": "层归一化操作。torch.nn.functional 接口。",
    "torch/nn/functional.py:scaled_dot_product_attention": "缩放点积注意力。FlashAttention 优先的高效注意力实现。",
    "torch/nn/parameter.py:Parameter": "可训练参数的张量子类。在 module.parameters() 中自动注册。",
    "torch/nn/parameter.py:UninitializedParameter": "未初始化的参数。支持延迟初始化。",

    # torch/fx/
    "torch/fx/graph.py:Graph": "FX 计算图。包含表示计算操作的有向无环节点集合。",
    "torch/fx/graph.py:GraphModule": "FX 图模块。包含计算图和前向执行方法。",
    "torch/fx/node.py:Node": "FX 图节点。表示计算图中的一个操作（如占位符、调用函数、返回值）。",
    "torch/fx/node.py:Argument": "FX 节点参数类型。可以是节点、值或嵌套的参数结构。",
    "torch/fx/node.py:map_aggregate": "递归遍历 FX 图中的嵌套参数结构。",
    "torch/fx/tracer.py:Tracer": "FX 符号追踪器。通过记录张量操作自动构建计算图。",
    "torch/fx/tracer.py:Proxy": "追踪代理。在追踪过程中包装真实张量以捕获操作。",
    "torch/fx/interpreter.py:Interpreter": "FX 图解释器。按拓扑顺序执行图中的节点。",
    "torch/fx/interpreter.py:Transformer": "FX 图变换器。遍历并转换图中的所有节点。",
    "torch/fx/subgraph_rewriter.py:replace_pattern": "FX 子图替换。用新模式替换图中匹配的子图模式。",

    # torch/_dynamo/
    "torch/_dynamo/eval_frame.py:optimize": "torch.compile 的优化装饰器。将函数编译为更快的实现。",
    "torch/_dynamo/convert_frame.py:convert_frame": "将 Python 帧编译为 FX 图的核心函数。",
    "torch/_dynamo/symbolic_convert.py:InstructionTranslator": "字节码指令转换器。将 Python 字节码指令转换为 FX 图操作。",
    "torch/_dynamo/guards.py:Guard": "动态形状保护。检查运行时条件以验证缓存的编译结果是否有效。",
    "torch/_dynamo/codegen.py:CodeGen": "Dynamo 代码生成器。从 FX 图生成可执行的 Python 代码。",

    # torch/_inductor/
    "torch/_inductor/compile_fx.py:compile_fx": "Inductor 编译入口。将 FX 图模块编译为优化的内核代码。",
    "torch/_inductor/graph.py:GraphLowering": "Inductor 图降级器。管理 IR 图的构建和调度。",
    "torch/_inductor/ir.py:IRNode": "Inductor 中间表示节点基类。所有 IR 操作的基类型。",
    "torch/_inductor/ir.py:TensorBox": "IR 张量容器。包装 IR 节点以提供类似张量的接口。",
    "torch/_inductor/lowering.py:lower": "将 FX 操作降级为 Inductor IR。决定操作是分解为更简单操作还是直接映射。",
    "torch/_inductor/select_algorithm.py:select_algorithm": "自动算法选择。为给定操作选择最佳内核实现。",
    "torch/_inductor/select_algorithm.py:autotune": "自动调优。通过基准测试选择最快的内核实现。",

    # torch/export/
    "torch/export/exported_program.py:ExportedProgram": "导出程序表示。包含已导出模型的计算图、参数和元数据。",
    "torch/export/dynamic_shapes.py:Dim": "动态形状维度。用于在导出时指定动态维度的约束和范围。",
    "torch/export/graph_module.py:GraphModule": "Export 图模块。封装已导出的计算图和状态。",
    "torch/export/unflatten.py:unflatten": "将平坦化的导出图恢复为嵌套模块层次结构。",

    # torch/distributed/
    "torch/distributed/distributed_c10d.py:init_process_group": "初始化分布式进程组。设置分布式训练的通信后端。",
    "torch/distributed/distributed_c10d.py:get_rank": "返回当前进程的全局 rank。",
    "torch/distributed/distributed_c10d.py:get_world_size": "返回分布式训练中的进程总数。",
    "torch/distributed/distributed_c10d.py:all_reduce": "全局规约操作。求和所有进程的张量并分发结果。",
    "torch/distributed/distributed_c10d.py:all_gather": "全局收集操作。收集所有进程的张量。",
    "torch/distributed/distributed_c10d.py:broadcast": "广播操作。将一个进程的张量发送到所有其他进程。",
    "torch/distributed/distributed_c10d.py:scatter": "分散操作。将一个进程的数据切分后分发。",
    "torch/distributed/distributed_c10d.py:reduce": "规约操作。将来自所有进程的张量聚合。",
    "torch/distributed/distributed_c10d.py:all_to_all": "全到全通信。每个进程都有数据发送到所有进程。",
    "torch/distributed/distributed_c10d.py:barrier": "屏障同步。所有进程在此等待直到全部到达。",
    "torch/distributed/distributed_c10d.py:send": "点对点发送操作。将张量发送到指定进程。",
    "torch/distributed/distributed_c10d.py:recv": "点对点接收操作。从指定进程接收张量。",
    "torch/distributed/distributed_c10d.py:reduce_scatter": "规约-分散组合操作。先规约再分散结果。",
    "torch/distributed/distributed_c10d.py:monitored_barrier": "可监控的屏障操作。支持超时检测的屏障。",
    "torch/distributed/distributed_c10d.py:all_gather_into_tensor": "全局收集到张量的操作。高效地收集所有进程的张量到一个大张量。",
    "torch/distributed/distributed_c10d.py:reduce_scatter_tensor": "规约分散张量操作。高效地规约然后分散。",
    "torch/distributed/distributed_c10d.py:ProcessGroup": "进程组类。管理分布式通信的进程组。",
    "torch/distributed/distributed_c10d.py:Backend": "通信后端枚举。支持 NCCL、Gloo 和 MPI。",

    # torchgen/
    "torchgen/gen.py:main": "TorchGen 主入口。读取 native_functions.yaml 并生成 C++ 代码。",
    "torchgen/gen.py:FileManager": "文件管理器。管理生成文件的写出。",
    "torchgen/gen.py:GroupedNativeFunctions": "分组的原生函数集合。按操作符名分组。",
    "torchgen/model.py:NativeFunction": "原生函数模型。表示一个来自 native_functions.yaml 的操作定义。",
    "torchgen/model.py:NativeFunctionsGroup": "原生函数组。包含函数及其变体（如 forward/backward）。",
    "torchgen/model.py:Schema": "操作模式。定义操作的类型签名。",
    "torchgen/model.py:FunctionSchema": "函数模式。定义函数的参数和返回值类型。",
    "torchgen/model.py:OperatorName": "操作符名称。代码生成中使用的操作符标识。",
    "torchgen/model.py:Tag": "操作符标签。分类和标记操作符（如 core、pointwise、view）。",
    "torchgen/dest/register_dispatch_key.py:gen": "生成调度键注册代码。生成 C++ 中将算子注册到指定后端（如 CUDA）的代码。",
    "torchgen/api/python.py:gen": "生成 Python 绑定代码。为 C++ 算子生成 Python API 绑定。",
    "torchgen/api/cpp.py:gen": "生成 C++ API 代码。为原生函数生成 C++ 接口声明。",

    # torch/sparse
    "torch/sparse/__init__.py:coalesce": "稀疏张量合并操作。合并重复索引并累加重复项的值。",

    # torch/optim
    "torch/optim/optimizer.py:Optimizer": "优化器基类。定义所有优化器的公共接口和参数组管理。",
    "torch/optim/optimizer.py:_grads": "梯度管理。在优化步骤中访问和操作参数的梯度。",
    "torch/optim/lr_scheduler.py:LRScheduler": "学习率调度器基类。提供学习率调整的通用框架。",
    "torch/optim/lr_scheduler.py:ReduceLROnPlateau": "自适应学习率调度器。当指标停止改善时降低学习率。",
    "torch/optim/sgd.py:SGD": "随机梯度下降优化器。支持动量、权重衰减和 Nesterov。",
    "torch/optim/adam.py:Adam": "Adam 优化器。自适应矩估计，结合动量和 RMSProp。",
    "torch/optim/adamw.py:AdamW": "AdamW 优化器。Adam 的解耦权重衰减变体。",

    # torch/utils/
    "torch/utils/_pytree.py:tree_map": "对 PyTree 结构的每个叶节点应用函数。",
    "torch/utils/_pytree.py:tree_flatten": "展平 PyTree 结构为叶节点列表和规格说明。",
    "torch/utils/_pytree.py:tree_unflatten": "根据规格说明重建 PyTree 结构。",

    # torch/_functorch
    "torch/_functorch/vmap.py:vmap": "自动向量化变换。将批处理维度映射到函数调用上。",
    "torch/_functorch/eager_transforms.py:grad": "梯度计算变换。返回计算梯度的函数。",
    "torch/_functorch/eager_transforms.py:vjp": "向量-雅可比积计算。用于反向模式自动微分。",
    "torch/_functorch/eager_transforms.py:jacrev": "雅可比矩阵计算（反向模式）。",
    "torch/_functorch/eager_transforms.py:jacfwd": "雅可比矩阵计算（前向模式）。",
    "torch/_functorch/eager_transforms.py:hessian": "海森矩阵计算。",

    # torch/jit
    "torch/jit/_script.py:script": "TorchScript 脚本模式。将函数编译为 TorchScript。",
    "torch/jit/_trace.py:trace": "TorchScript 追踪模式。通过示例输入追踪构建 TorchScript。",
    "torch/jit/_trace.py:trace_module": "追踪整个模块。递归追踪模块的所有方法。",

    # torch/onnx
    "torch/onnx/__init__.py:export": "ONNX 导出。将 PyTorch 模型导出为 ONNX 格式。",
    "torch/onnx/__init__.py:register_custom_op": "注册自定义 ONNX 操作符映射。",

    # torch/_subclasses
    "torch/_subclasses/fake_tensor.py:FakeTensor": "假张量。没有真实数据但携带元数据（形状、dtype、设备）的张量。用于元数据传播而不执行计算。",
    "torch/_subclasses/fake_tensor.py:MetaConverter": "元数据转换器。在真实张量和假张量之间转换。",
    "torch/_subclasses/functional_tensor.py:FunctionalTensor": "函数式张量。支持原地操作的函数式表示。",
}

# ── Apply function/class descriptions ──
print("=== 1. Enhancing function/class nodes ===")
enhanced_fns = 0
for n in nodes:
    nid = n["id"]
    if n["type"] == "function":
        # Extract just the function name for matching
        parts = nid.split(":")
        if len(parts) >= 3:
            key = ":".join(parts[1:])  # e.g., "torch/_tensor.py:Tensor.size"
            if key in FUNCTION_DESCRIPTIONS:
                n["summary"] = FUNCTION_DESCRIPTIONS[key]
                # Add better tags
                if "tags" in n and n["tags"] == ["function"]:
                    if any(kw in key.lower() for kw in ["__add__", "__mul__", "__matmul__", "__getitem__", "operator"]):
                        n["tags"] = ["function", "operator-overload", "tensor-operation"]
                    elif "init" in key.lower() or "uniform_" in key.lower() or "normal_" in key.lower():
                        n["tags"] = ["function", "initialization", "neural-network"]
                    elif "loss" in key.lower():
                        n["tags"] = ["function", "loss", "neural-network"]
                    elif "autograd" in key.lower() or "backward" in key.lower() or "grad" in key.split(":")[-1].lower():
                        n["tags"] = ["function", "autograd", "differentiation"]
                    elif "cuda" in key.lower() or "cuda:" in key:
                        n["tags"] = ["function", "cuda", "gpu"]
                    else:
                        n["tags"] = ["function"]
                enhanced_fns += 1

print(f"  Enhanced {enhanced_fns} function nodes")

# ── Also enhance class nodes ──
enhanced_cls = 0
for n in nodes:
    nid = n["id"]
    if n["type"] == "class":
        parts = nid.split(":")
        if len(parts) >= 3:
            key = ":".join(parts[1:])
            if key in FUNCTION_DESCRIPTIONS:  # reuse same dict for classes
                n["summary"] = FUNCTION_DESCRIPTIONS[key]
                enhanced_cls += 1

print(f"  Enhanced {enhanced_cls} class nodes")

# ── 2. Add inherits edges for known class hierarchies ──
print("\n=== 2. Adding inherits edges ===")
inheritance_hierarchy = [
    ("class:torch/nn/modules/container.py:Module", None),  # root
    ("class:torch/nn/modules/linear.py:Linear", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/conv.py:Conv1d", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/conv.py:Conv2d", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/rnn.py:LSTM", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/rnn.py:GRU", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/transformer.py:Transformer", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/activation.py:ReLU", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/activation.py:GELU", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/normalization.py:LayerNorm", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/normalization.py:BatchNorm1d", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/normalization.py:BatchNorm2d", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/pooling.py:MaxPool2d", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/pooling.py:AvgPool2d", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/container.py:Sequential", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/container.py:ModuleList", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/container.py:ModuleDict", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/loss.py:CrossEntropyLoss", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/loss.py:MSELoss", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/dropout.py:Dropout", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/batchnorm.py:BatchNorm1d", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/batchnorm.py:BatchNorm2d", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/adaptive.py:AdaptiveAvgPool2d", "class:torch/nn/modules/container.py:Module"),
    ("class:torch/nn/modules/padding.py:ReflectionPad2d", "class:torch/nn/modules/container.py:Module"),
    # Autograd
    ("class:torch/autograd/function.py:Function", None),
    ("class:torch/autograd/graph.py:Node", None),
    # Optimizer
    ("class:torch/optim/optimizer.py:Optimizer", None),
    ("class:torch/optim/sgd.py:SGD", "class:torch/optim/optimizer.py:Optimizer"),
    ("class:torch/optim/adam.py:Adam", "class:torch/optim/optimizer.py:Optimizer"),
    ("class:torch/optim/adamw.py:AdamW", "class:torch/optim/optimizer.py:Optimizer"),
    ("class:torch/optim/adagrad.py:Adagrad", "class:torch/optim/optimizer.py:Optimizer"),
    ("class:torch/optim/rmsprop.py:RMSprop", "class:torch/optim/optimizer.py:Optimizer"),
    ("class:torch/optim/lr_scheduler.py:LRScheduler", None),
    # FX
    ("class:torch/fx/graph.py:Graph", None),
    ("class:torch/fx/graph.py:GraphModule", None),
    # Dynamo
    ("class:torch/_dynamo/symbolic_convert.py:InstructionTranslator", None),
    # Inductor
    ("class:torch/_inductor/ir.py:IRNode", None),
    # Tensor
    ("class:torch/_subclasses/fake_tensor.py:FakeTensor", None),
    ("class:torch/_subclasses/functional_tensor.py:FunctionalTensor", None),
]

# Find Module root
existing_inherits = set((e["source"], e["target"]) for e in edges if e["type"] == "inherits")
inherits_added = 0
for child, parent in inheritance_hierarchy:
    if child not in node_map:
        continue
    if parent is None:
        continue
    if parent not in node_map:
        continue
    if (child, parent) in existing_inherits:
        continue
    edges.append({
        "source": child,
        "target": parent,
        "type": "inherits",
        "direction": "forward",
        "weight": 0.9,
    })
    inherits_added += 1

print(f"  Added {inherits_added} inherits edges")

# ── 3. Add exports edges for function/class nodes that are exported ──
print("\n=== 3. Adding exports edges ===")
# Check __all__ or public API patterns
existing_exports = set((e["source"], e["target"]) for e in edges if e["type"] == "exports")
exports_added = 0

# For key files, mark their top-level content as exported
export_sources = {  # file → list of function/class IDs
    "file:torch/__init__.py": None,  # everything is exported
}

for n in nodes:
    if n["type"] in ("function", "class") and "filePath" in n:
        fp = n["filePath"]
        fid = f"file:{fp}"
        if fid in export_sources and fid in node_map:
            if (fid, n["id"]) not in existing_exports:
                edges.append({
                    "source": fid,
                    "target": n["id"],
                    "type": "exports",
                    "direction": "forward",
                    "weight": 0.8,
                })
                exports_added += 1

print(f"  Added {exports_added} exports edges")

# ── 4. Add tested_by edges for known test relationships ──
print("\n=== 4. Adding tested_by edges ===")
existing_tested = set((e["source"], e["target"]) for e in edges if e["type"] == "tested_by")
tested_added = 0

# Map test files to their production files
test_mappings = {
    "file:torch/testing/_internal/common_utils.py": ["file:torch/__init__.py", "file:torch/_tensor.py"],
}
for test_file, prod_files in test_mappings.items():
    if test_file not in node_map:
        continue
    for pf in prod_files:
        if pf not in node_map:
            continue
        if (pf, test_file) not in existing_tested:
            edges.append({
                "source": pf,
                "target": test_file,
                "type": "tested_by",
                "direction": "forward",
                "weight": 0.5,
            })
            tested_added += 1

print(f"  Added {tested_added} tested_by edges")

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
