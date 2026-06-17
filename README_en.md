![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------

PyTorch 是一个 Python 软件包，提供以下两个高级功能：
- 具有强大 GPU 加速的张量计算（类似 NumPy）
- 基于磁带式自动求导系统的深度神经网络

你可以复用你喜欢的 Python 软件包（如 NumPy、SciPy 和 Cython），在需要时扩展 PyTorch。

我们的主干健康状态（持续集成信号）可在 [hud.pytorch.org](https://hud.pytorch.org/ci/pytorch/pytorch/main) 查看。

<!-- toc -->

- [更多关于 PyTorch](#更多关于-pytorch)
  - [GPU 就绪的张量库](#gpu-就绪的张量库)
  - [动态神经网络：磁带式自动求导](#动态神经网络磁带式自动求导)
  - [Python 优先](#python-优先)
  - [命令式体验](#命令式体验)
  - [快速且轻量](#快速且轻量)
  - [无痛扩展](#无痛扩展)
- [安装](#安装)
  - [二进制包](#二进制包)
    - [NVIDIA Jetson 平台](#nvidia-jetson-平台)
  - [从源码构建](#从源码构建)
    - [前提条件](#前提条件)
      - [NVIDIA CUDA 支持](#nvidia-cuda-支持)
      - [AMD ROCm 支持](#amd-rocm-支持)
      - [Intel GPU 支持](#intel-gpu-支持)
    - [获取 PyTorch 源码](#获取-pytorch-源码)
    - [安装依赖](#安装依赖)
    - [安装 PyTorch](#安装-pytorch)
      - [调整构建选项（可选）](#调整构建选项可选)
  - [Docker 镜像](#docker-镜像)
    - [使用预构建镜像](#使用预构建镜像)
    - [自行构建镜像](#自行构建镜像)
  - [构建文档](#构建文档)
    - [CI 错误排查](#ci-错误排查)
    - [构建 PDF](#构建-pdf)
  - [历史版本](#历史版本)
- [快速入门](#快速入门)
- [资源](#资源)
- [交流](#交流)
- [发布与贡献](#发布与贡献)
- [团队](#团队)
- [许可证](#许可证)

<!-- tocstop -->

## 更多关于 PyTorch

[学习 PyTorch 基础知识](https://pytorch.org/tutorials/beginner/basics/intro.html)

从细粒度来看，PyTorch 是一个由以下组件构成的库：

| 组件 | 描述 |
| ---- | --- |
| [**torch**](https://pytorch.org/docs/stable/torch.html) | 类似 NumPy 的张量库，具有强大的 GPU 支持 |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | 基于磁带式的自动微分库，支持 torch 中所有可微分张量操作 |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html) | 编译栈（TorchScript），用于从 PyTorch 代码创建可序列化和可优化的模型 |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html) | 与 autograd 深度集成的神经网络库，设计上具有最大灵活性 |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python 多进程，但跨进程共享 torch 张量的内存。适用于数据加载和 Hogwild 训练 |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html) | DataLoader 及其他实用工具函数 |

通常，PyTorch 有以下两种使用方式：

- 作为 NumPy 的替代品，利用 GPU 的计算能力。
- 作为深度学习研究平台，提供最大的灵活性和速度。

进一步阐述：

### GPU 就绪的张量库

如果你使用 NumPy，那么你已经在使用张量（即 ndarray）。

![张量图示](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/tensor_illustration.png)

PyTorch 提供的张量可以驻留在 CPU 或 GPU 上，并能大幅加速计算。

我们提供各种各样的张量例程来加速和满足你的科学计算需求，如切片、索引、数学运算、线性代数、归约等。而且它们非常快！

### 动态神经网络：磁带式自动求导

PyTorch 有一种独特的构建神经网络的方式：使用和重放磁带记录器。

大多数框架（如 TensorFlow、Theano、Caffe 和 CNTK）对世界有静态的视角。你必须先构建神经网络，然后反复复用相同的结构。改变网络行为意味着必须从头开始。

使用 PyTorch，我们采用了一种称为反向模式自动微分的（reverse-mode auto-differentiation）技术，它允许你以零延迟或零开销的方式任意改变网络行为。我们的灵感来自关于此主题的多篇研究论文，以及当前和过去的工作，如 [torch-autograd](https://github.com/twitter/torch-autograd)、[autograd](https://github.com/HIPS/autograd)、[Chainer](https://chainer.org) 等。

虽然这种技术并非 PyTorch 独有，但它是目前最快的实现之一。你可以在疯狂的研究中获得速度和灵活性的最佳结合。

![动态图](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

### Python 优先

PyTorch 不是一个单体 C++ 框架的 Python 绑定。它被构建为深度集成到 Python 中。你可以像使用 [NumPy](https://www.numpy.org/) / [SciPy](https://www.scipy.org/) / [scikit-learn](https://scikit-learn.org) 一样自然地使用它。你可以用 Python 本身编写新的神经网络层，使用你喜欢的库和 [Cython](https://cython.org/)、[Numba](http://numba.pydata.org/) 等软件包。我们的目标是在合适的地方不重复造轮子。

### 命令式体验

PyTorch 设计为直观、线性思维和易于使用。当你执行一行代码时，它就会被执行。不存在异步的世界视图。当你进入调试器或收到错误信息和堆栈跟踪时，理解它们是直截了当的。堆栈跟踪精确指向你定义代码的位置。我们希望你不会因为糟糕的堆栈跟踪或异步和不透明的执行引擎而花费数小时调试代码。

### 快速且轻量

PyTorch 具有最小的框架开销。我们集成了加速库，如 [Intel MKL](https://software.intel.com/mkl) 和 NVIDIA（[cuDNN](https://developer.nvidia.com/cudnn)、[NCCL](https://developer.nvidia.com/nccl)）以最大化速度。在其核心，其 CPU 和 GPU 张量与神经网络后端是成熟的，并经过了多年的测试。

因此，无论你运行小型还是大型神经网络，PyTorch 都相当快。

与其他一些替代方案相比，PyTorch 中的内存使用极为高效。我们为 GPU 编写了自定义内存分配器，以确保你的深度学习模型具有最高的内存效率。这使你能够训练比以前更大的深度学习模型。

### 无痛扩展

编写新的神经网络模块，或与 PyTorch 的张量 API 交互，设计为直截了当且抽象最少。

你可以使用 torch API [或你喜欢的基于 NumPy 的库（如 SciPy）](https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html)用 Python 编写新的神经网络层。

如果你想用 C/C++ 编写层，我们提供了方便的扩展 API，高效且样板代码最少。无需编写包装代码。你可以参见[这里的教程](https://pytorch.org/tutorials/advanced/cpp_extension.html)和[这里的示例](https://github.com/pytorch/extension-cpp)。


## 安装

### 二进制包

通过 Conda 或 pip wheel 安装二进制包的命令见我们的网站：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)


#### NVIDIA Jetson 平台

适用于 NVIDIA Jetson Nano、Jetson TX1/TX2、Jetson Xavier NX/AGX 和 Jetson AGX Orin 的 Python wheels 在[这里](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048)提供，L4T 容器在[这里](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)发布。

它们需要 JetPack 4.2 及以上版本，由 [@dusty-nv](https://github.com/dusty-nv) 和 [@ptrblck](https://github.com/ptrblck) 维护。


### 从源码构建

#### 前提条件
如果你从源码安装，你需要：
- Python 3.10 或更高版本
- 一个完全支持 C++20 的编译器，如 clang 或 gcc（Linux 上需要 gcc 11.3.0 或更新版本）
- Visual Studio 或 Visual Studio Build Tool（仅 Windows）
- 至少 10 GB 的可用磁盘空间
- 首次构建需 30-60 分钟（后续重新构建会快得多）

\* PyTorch CI 使用 Visual C++ BuildTools，这些工具随 Visual Studio Enterprise、Professional 或 Community 版本提供。你也可以从 https://visualstudio.microsoft.com/visual-cpp-build-tools/ 安装构建工具。默认情况下，构建工具*不*随 Visual Studio Code 提供。

环境设置示例如下：

* Linux：

```bash
$ source <CONDA_INSTALL_DIR>/bin/activate
$ conda create -y -n <CONDA_NAME>
$ conda activate <CONDA_NAME>
```
* Windows：

```bash
$ source <CONDA_INSTALL_DIR>\Scripts\activate.bat
$ conda create -y -n <CONDA_NAME>
$ conda activate <CONDA_NAME>
$ call "C:\Program Files\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
```

Conda 环境不是必需的。你也可以在标准虚拟环境中进行 PyTorch 构建，例如使用 `uv` 等工具创建的虚拟环境，前提是你的系统已安装所有必要的无法通过 pip 获取的依赖（如 CUDA、MKL）。

##### NVIDIA CUDA 支持
如果你想编译 CUDA 支持，[从我们的支持矩阵中选择一个受支持的 CUDA 版本](https://pytorch.org/get-started/locally/)，然后安装以下内容：
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v9.0 或更高版本
- 与 CUDA [兼容的编译器](https://gist.github.com/ax3l/9489132)

注意：你可以参考 [cuDNN 支持矩阵](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) 了解不同 CUDA、CUDA 驱动和 NVIDIA 硬件支持的各种 cuDNN 版本。

如果你想禁用 CUDA 支持，导出环境变量 `USE_CUDA=0`。其他可能有用的环境变量可在 `setup.py` 中找到。如果 CUDA 安装在非标准位置，请设置 PATH 以便找到要使用的 nvcc（例如 `export PATH=/usr/local/cuda-12.8/bin:$PATH`）。

如果你正在为 NVIDIA 的 Jetson 平台（Jetson Nano、TX1、TX2、AGX Xavier）进行构建，安装 PyTorch for Jetson Nano 的说明见[这里](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

##### AMD ROCm 支持
如果你想编译 ROCm 支持，请安装：
- [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 及以上版本
- ROCm 目前仅支持 Linux 系统。

默认情况下，构建系统期望 ROCm 安装在 `/opt/rocm` 中。如果 ROCm 安装在其他目录中，必须将 `ROCM_PATH` 环境变量设置为 ROCm 安装目录。构建系统会自动检测 AMD GPU 架构。可选地，可以通过 `PYTORCH_ROCM_ARCH` 环境变量显式设置 AMD GPU 架构 [AMD GPU 架构](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

如果你想禁用 ROCm 支持，导出环境变量 `USE_ROCM=0`。其他可能有用的环境变量可在 `setup.py` 中找到。

##### Intel GPU 支持
如果你想编译 Intel GPU 支持，请遵循以下说明：
- [PyTorch Intel GPU 前提条件](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html) 说明。
- Intel GPU 支持 Linux 和 Windows。

如果你想禁用 Intel GPU 支持，导出环境变量 `USE_XPU=0`。其他可能有用的环境变量可在 `setup.py` 中找到。

#### 获取 PyTorch 源码

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
# 如果你正在更新现有的检出
git submodule sync
git submodule update --init --recursive
```

#### 安装依赖

**通用**

```bash
# 在使用上述"获取 PyTorch 源码"部分克隆源代码后，从 PyTorch 目录运行此命令
pip install --group dev
```

**Linux**

```bash
pip install mkl-static mkl-include
# 仅 CUDA：如果需要，为 GPU 添加 LAPACK 支持
# magma 安装：在激活 conda 环境下运行。指定要安装的 CUDA 版本
.ci/docker/common/install_magma_conda.sh 12.4

# （可选）如果使用 torch.compile 与 inductor/triton，安装匹配版本的 triton
# 克隆后在 pytorch 目录中运行
# 对于 Intel GPU 支持，在运行命令前请显式 `export USE_XPU=1`。
make triton
```

**Windows**

```bash
pip install mkl-static mkl-include
# 如果需要 torch.distributed，添加这些软件包。
# Windows 上的分布式包支持是原型功能，可能会有变化。
conda install -c conda-forge libuv=1.51
```

#### 安装 PyTorch

**Linux**

如果你正在为 AMD ROCm 编译，首先运行此命令：

```bash
# 仅在你正在为 ROCm 编译时运行
python tools/amd_build/build_amd.py
```

安装 PyTorch

```bash
# conda 环境的 CMake 前缀
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install --no-build-isolation -v -e .

# 非 conda 环境（如 Python venv）的 CMake 前缀
# 在激活 venv 后调用以下命令
export CMAKE_PREFIX_PATH="${VIRTUAL_ENV}:${CMAKE_PREFIX_PATH}"
```

**macOS**

```bash
python -m pip install --no-build-isolation -v -e .
```

**Windows**

如果你想构建传统 Python 代码，请参考 [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**仅 CPU 构建**

在此模式下，PyTorch 计算将在你的 CPU 上运行，而不是 GPU。

```cmd
python -m pip install --no-build-isolation -v -e .
```

关于 OpenMP 的说明：期望的 OpenMP 实现是 Intel OpenMP（iomp）。为了链接 iomp，你需要手动下载库，并通过调整 `CMAKE_INCLUDE_PATH` 和 `LIB` 设置构建环境。[这里](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.rst#building-from-source)的说明是设置 MKL 和 Intel OpenMP 的示例。如果没有这些 CMake 配置，将使用 Microsoft Visual C OpenMP 运行时（vcomp）。

**基于 CUDA 的构建**

在此模式下，PyTorch 计算将通过 CUDA 利用你的 GPU 进行更快的数值计算。

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) 是构建支持 CUDA 的 PyTorch 所必需的。NVTX 是 CUDA 发行版的一部分，在其中被称为"Nsight Compute"。要将其安装到已安装的 CUDA 上，请再次运行 CUDA 安装并选中相应的复选框。确保在 Visual Studio 之后安装带有 Nsight Compute 的 CUDA。

目前，VS 2017 / 2019 和 Ninja 被支持作为 CMake 的生成器。如果在 `PATH` 中检测到 `ninja.exe`，则 Ninja 将作为默认生成器，否则将使用 VS 2017 / 2019。<br/> 如果选择 Ninja 作为生成器，将选择最新的 MSVC 作为底层工具链。

通常需要额外的库，如 [Magma](https://developer.nvidia.com/magma)、[oneDNN（即 MKLDNN 或 DNNL）](https://github.com/oneapi-src/oneDNN) 和 [Sccache](https://github.com/mozilla/sccache)。请参考 [installation-helper](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers) 来安装它们。

你可以参考 [build_pytorch.bat](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/win-test-helpers/build_pytorch.bat) 脚本了解其他环境变量配置。

```cmd
cmd

:: 在下载并解压 mkl 包后设置环境变量，
:: 否则 CMake 会抛出 `Could NOT find OpenMP` 错误。
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%

:: 在继续之前，请仔细阅读前面部分的内容。
:: [可选] 如果你想覆盖 Ninja 和 Visual Studio 与 CUDA 使用的底层工具集，请运行以下脚本块。
:: "Visual Studio 2019 Developer Command Prompt" 将自动运行。
:: 使用 Visual Studio 生成器时，确保 CMake >= 3.12。
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

:: [可选] 如果你想覆盖 CUDA 宿主编译器
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe

python -m pip install --no-build-isolation -v -e .
```

**Intel GPU 构建**

在此模式下，将构建支持 Intel GPU 的 PyTorch。

请确保[通用前提条件](#前提条件)以及 [Intel GPU 的前提条件](#intel-gpu-支持)已正确安装，并在开始构建之前配置好环境变量。对于构建工具，需要 `Visual Studio 2022`。

然后可以使用以下命令构建 PyTorch：

```cmd
:: CMD 命令：
:: 设置 CMAKE_PREFIX_PATH 以帮助找到相应的包
:: %CONDA_PREFIX% 仅在 `conda activate custom_env` 之后有效

if defined CMAKE_PREFIX_PATH (
    set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library;%CMAKE_PREFIX_PATH%"
) else (
    set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library"
)

python -m pip install --no-build-isolation -v -e .
```

#### 调整构建选项（可选）

你可以通过以下方式可选地调整 cmake 变量的配置（无需先构建）。例如，调整预先检测到的 CuDNN 或 BLAS 目录可以通过此步骤完成。

Linux

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CMAKE_ONLY=1 python setup.py build
ccmake build  # 或 cmake-gui build
```

macOS

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
MACOSX_DEPLOYMENT_TARGET=11.0 CMAKE_ONLY=1 python setup.py build
ccmake build  # 或 cmake-gui build
```

### Docker 镜像

#### 使用预构建镜像

你也可以从 Docker Hub 拉取预构建的 docker 镜像，并使用 docker v23.0+ 运行

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

请注意，PyTorch 使用共享内存在进程之间共享数据，因此如果使用 torch 多进程（例如多线程数据加载器），容器运行的默认共享内存段大小不足，你应该通过 `--ipc=host` 或 `--shm-size` 命令行选项增加共享内存大小。

#### 自行构建镜像

**注意：** 必须使用 Docker 版本 >= 23.0 构建

Dockerfile 用于构建支持 CUDA 12.1 和 cuDNN v9 的镜像。你可以传递 `PYTHON_VERSION=x.y` make 变量来指定 Miniconda 使用的 Python 版本，或者不设置它使用默认值，因为 Dockerfile 使用系统 Python。

```bash
make -f docker.Makefile
# 镜像被标记为 docker.io/${your_docker_username}/pytorch
```

你也可以传递 `CMAKE_VARS="..."` 环境变量来指定在构建期间传递给 CMake 的额外 CMake 变量。参见 [setup.py](./setup.py) 了解可用变量的列表。

```bash
make -f docker.Makefile
```

### 构建文档

要以各种格式构建文档，你需要 [Sphinx](http://www.sphinx-doc.org) 和 `pytorch_sphinx_theme2`。

在本地构建文档之前，请确保你的环境中已安装 `torch`。对于小的修复，可以按照[快速入门](https://pytorch.org/get-started/locally/)中所述安装 nightly 版本。

对于更复杂的修复，例如添加新模块和新模块的 docstring，你可能需要[从源码](#从源码构建)安装 torch。参见 [Docstring Guidelines](https://github.com/pytorch/pytorch/wiki/Docstring-Guidelines) 了解 docstring 约定。

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

运行 `make` 获取所有可用输出格式的列表。

如果你遇到 katex 错误，运行 `npm install katex`。如果问题仍然存在，尝试 `npm install -g katex`

> [!NOTE]
> 如果你看到 numpy 不兼容错误，运行：
> ```
> pip install 'numpy<2'
> ```


#### CI 错误排查
你的构建可能会显示本地没有的错误——以下是如何找到与文档相关的错误。

如果构建有任何错误，你将在 PR 上看到类似这样的信息：

<img width="781" height="400" alt="Monosnap Update installation instructions for doc build · Pull Request #169534 · pytorch:pytorch 2025-12-18 18-22-53" src="https://github.com/user-attachments/assets/49a3dfe7-81c2-4246-852b-bc3f807e95af" />

任何与文档相关的错误都会出现在标题中某处包含"doc"的作业中。看起来这些作业中没有一个与我们的文档相关。

无论如何，让我们看一下。点击作业查看日志：

<img width="1187" height="668" alt="Monosnap Update installation instructions for doc build · pytorch:pytorch@7380336 2025-12-18 18-24-15" src="https://github.com/user-attachments/assets/117df543-8356-4323-8e1c-ef02a95554ba" />

我们可以确定此作业不涉及文档。

查看此构建，我们可以看到这些作业与我们的文档相关——并且它们没有任何错误：

<img width="777" height="395" alt="Check the docs jobs" src="https://github.com/user-attachments/assets/5d7c196b-2d40-49ad-87e3-f57de6e14a5b" />

你可能还会在 PR 上看到这样的评论：

<img width="651" height="246" alt="PR Comment" src="https://github.com/user-attachments/assets/27e0120a-ba33-4b1c-b4a5-bf3064520586" />

我们可以看到其中一些问题与我们的文档相关。

通过点击 `gh` 链接打开日志：

<img width="873" height="360" alt="View Logs" src="https://github.com/user-attachments/assets/ab5b862f-8026-489c-b95e-a6cd4257e4b7" />

在这里我们可以看到有一个与文档相关的错误：

<img width="1117" height="433" alt="Doc Error" src="https://github.com/user-attachments/assets/0a275921-736d-43a7-ab0f-3e8854d43280" />

你总是可以通过转到 PR 上的 `Checks` 标签页，并向下滚动到 `pull` 来找到相关的文档构建。

<img width="481" height="561" alt="checks" src="https://github.com/user-attachments/assets/eef18f2b-7134-4e2e-bd90-bcdc12800132" />

你可以点击进入或切换折叠面板查看所有作业，在这里你可以看到高亮的文档作业：

<img width="570" height="611" alt="jobs" src="https://github.com/user-attachments/assets/f62812ca-caee-421b-863c-54f38fd28d46" />

如果你点击进入，你将在底部看到文档作业，如下所示：

<img width="354" height="312" alt="View Docs jobs" src="https://github.com/user-attachments/assets/8fadb935-5314-4c4b-a1b5-133781754f03" />

#### 构建 PDF

要编译所有 PyTorch 文档的 PDF，请确保你已安装 `texlive` 和 LaTeX。在 macOS 上，你可以使用以下命令安装它们：

```
brew install --cask mactex
```

创建 PDF：

1. 运行：

   ```
   make latexpdf
   ```

   这将在 `build/latex` 目录中生成必要的文件。

2. 导航到此目录并执行：

   ```
   make LATEXOPTS="-interaction=nonstopmode"
   ```

   这将生成包含所需内容的 `pytorch.pdf`。再次运行此命令，使其生成正确的目录和索引。

> [!NOTE]
> 要查看目录，请在 PDF 查看器中切换到 **Table of Contents** 视图。

### 历史版本

历史版本的安装说明和二进制文件可在[我们的网站](https://pytorch.org/get-started/previous-versions)上找到。


## 快速入门

帮助你入门的一些指引：
- [教程：帮助你开始理解和使用 PyTorch](https://pytorch.org/tutorials/)
- [示例：涵盖所有领域的易于理解的 PyTorch 代码](https://github.com/pytorch/examples)
- [API 参考](https://pytorch.org/docs/)
- [术语表](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

## 资源

* [PyTorch.org](https://pytorch.org/)
* [PyTorch 教程](https://pytorch.org/tutorials/)
* [PyTorch 示例](https://github.com/pytorch/examples)
* [PyTorch 模型](https://pytorch.org/hub/)
* [Udacity 深度学习入门（PyTorch）](https://www.udacity.com/course/deep-learning-pytorch--ud188)
* [Udacity 机器学习入门（PyTorch）](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)
* [Coursera 深度神经网络（PyTorch）](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
* [PyTorch Twitter](https://twitter.com/PyTorch)
* [PyTorch 博客](https://pytorch.org/blog/)
* [PyTorch YouTube](https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw)

## 交流
* 论坛：讨论实现、研究等。https://discuss.pytorch.org
* GitHub Issues：Bug 报告、功能请求、安装问题、RFC、想法等。
* Slack：[PyTorch Slack](https://pytorch.slack.com/) 主要面向中高级 PyTorch 用户和开发者，用于一般聊天、在线讨论、协作等。如果你是需要帮助的初学者，主要渠道是 [PyTorch 论坛](https://discuss.pytorch.org)。如果你需要 Slack 邀请，请填写此表单：https://goo.gl/forms/PP1AGvNHpSaJP8to1
* 通讯：无噪音，单向电子邮件通讯，包含关于 PyTorch 的重要公告。你可以在此注册：https://eepurl.com/cbG0rv
* Facebook 页面：关于 PyTorch 的重要公告。https://www.facebook.com/pytorch
* 有关品牌指南，请访问我们的网站 [pytorch.org](https://pytorch.org/)

## 发布与贡献

通常情况下，PyTorch 每年有三个小版本。如果你遇到 bug，请通过[提交 issue](https://github.com/pytorch/pytorch/issues)告诉我们。

我们感谢所有贡献。如果你计划贡献 bug 修复，请直接进行，无需进一步讨论。

如果你计划贡献新功能、实用函数或对核心的扩展，请首先提交一个 issue 并与我们讨论该功能。在没有讨论的情况下发送 PR 可能会最终导致 PR 被拒绝，因为我们可能在核心方面朝着与你所知不同的方向发展。

要了解更多关于为 PyTorch 做贡献的信息，请参阅我们的[贡献页面](CONTRIBUTING.md)。有关 PyTorch 发布的更多信息，请参阅[发布页面](RELEASE.md)。

## 团队

PyTorch 是一个社区驱动的项目，有许多有能力的工程师和研究人员为其做出贡献。

PyTorch 目前由 [Soumith Chintala](http://soumith.ch)、[Gregory Chanan](https://github.com/gchanan)、[Dmytro Dzhulgakov](https://github.com/dzhulgakov)、[Edward Yang](https://github.com/ezyang)、[Alban Desmaison](https://github.com/albanD)、[Piotr Bialecki](https://github.com/ptrblck) 和 [Nikita Shulga](https://github.com/malfet) 维护，主要贡献来自数百位有才华的个人，以各种形式和方式。
一个非穷举但不断增长的名单需要提及：[Trevor Killeen](https://github.com/killeent)、[Sasank Chilamkurthy](https://github.com/chsasank)、[Sergey Zagoruyko](https://github.com/szagoruyko)、[Adam Lerer](https://github.com/adamlerer)、[Francisco Massa](https://github.com/fmassa)、[Alykhan Tejani](https://github.com/alykhantejani)、[Luca Antiga](https://github.com/lantiga)、[Alban Desmaison](https://github.com/albanD)、[Andreas Koepf](https://github.com/andreaskoepf)、[James Bradbury](https://github.com/jekbradbury)、[Zeming Lin](https://github.com/ebetica)、[Yuandong Tian](https://github.com/yuandong-tian)、[Guillaume Lample](https://github.com/glample)、[Marat Dukhan](https://github.com/Maratyszcza)、[Natalia Gimelshein](https://github.com/ngimel)、[Christian Sarofeen](https://github.com/csarofeen)、[Martin Raison](https://github.com/martinraison)、[Edward Yang](https://github.com/ezyang)、[Zachary Devito](https://github.com/zdevito)。 <!-- codespell:ignore -->

注意：此项目与同名的 [hughperkins/pytorch](https://github.com/hughperkins/pytorch) 无关。Hugh 是 Torch 社区的宝贵贡献者，并在许多 Torch 和 PyTorch 相关事务中提供了帮助。

## 许可证

PyTorch 采用 BSD 风格许可证，详见 [LICENSE](LICENSE) 文件。
