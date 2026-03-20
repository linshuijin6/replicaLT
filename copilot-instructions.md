# 说明（项目核心指令入口）

本仓库的项目级 Copilot 核心指令已迁移到：
- `.github/copilot-instructions.md`

该文件聚焦 MRI->PET + plasma 两阶段主链（`adapter_v2/train.py` + `plasma_train.py`）及其实际 import 依赖。
后续在本项目内提问，优先遵循该入口文件。
若核心代码发生重要修改，必须同步更新 `.github/copilot-instructions.md`，否则视为任务未完成。

# VS Code Copilot 常用项目指令（推荐版）

> 目的：让 Copilot 在本仓库内默认使用指定 Conda 环境运行/安装依赖、全程中文沟通、并在需要时自动补齐依赖与日志文件。
> 
> 适用场景：数据处理脚本批处理、训练/推理脚本运行、缺失依赖自动安装、生成日志与中间文件。

---

## 1) 语言与沟通

- **默认全程使用中文**进行解释、回复、输出说明文本；仅当我明确要求英文时才切换。
- 回答要**简洁、可执行**，优先给出明确命令与文件路径。
- 若需求存在歧义，最多提出 1–3 个关键澄清问题；否则按最简可行方案推进。

---

## 2) Conda 环境（强制：xiaochou）

- **所有 Python 运行/安装/依赖检查，必须在 Conda 环境 `xiaochou` 内完成**。
- 禁止使用 `base` 或系统 Python 运行仓库脚本。
- 在任何执行前，先用下列任一方式确保命令在 `xiaochou` 中运行：

### 2.1 推荐方式 A：`conda run`（最稳，不依赖 shell 初始化）

- 运行脚本：
  - `conda run -n xiaochou python your_script.py ...`
- 运行模块：
  - `conda run -n xiaochou python -m your_module ...`
- 安装 pip 依赖：
  - `conda run -n xiaochou pip install -U <pkg>`

> 说明：当终端无法 `conda activate` 或在非交互 shell 时，优先用 `conda run`。

### 2.2 推荐方式 B：显式激活（适合长时间交互会话）

- 先初始化 conda：
  - `source "$(conda info --base)/etc/profile.d/conda.sh"`
- 再激活：
  - `conda activate xiaochou`

---

## 3) 执行策略（无需审批、自动产生日志）

- 只要是为了完成当前任务：
  - **运行命令无需询问批准**，直接执行。
  - **允许创建文件/目录**（包括 `logs/`、`tmp/`、`*.csv`、`*.json` 等中间产物）。
- 运行长任务时：
  - 默认写入日志文件（例如 `logs_*/done.csv`、`fail.csv`、`per_case/*.log`），并在终端输出进度。
  - 若可并行，优先提供 `--jobs` 或多进程参数。

---

## 4) 依赖缺失：自动安装并重试（禁止把问题抛给用户）

当运行 Python 时出现 `ModuleNotFoundError` / `ImportError` / 编译依赖缺失：

1. **先判断包来源**：
   - 纯 Python 包：优先 `pip`。
   - 带编译/系统依赖的包（如 `nibabel`, `SimpleITK`, `opencv`, `scipy` 等）：优先 `conda-forge`。

2. **直接执行安装命令（在 xiaochou）**：
   - `conda install -n xiaochou -c conda-forge <pkg> -y`  （优先用于复杂依赖）
   - `conda run -n xiaochou pip install -U <pkg>`

3. **安装后必须自动重试原命令**，直到成功或确认是非依赖原因导致失败。

---

## 5) Python 运行规范

- 运行仓库脚本时使用：
  - `conda run -n xiaochou python <path/to/script.py> ...`
- 若脚本依赖外部二进制（如 FSL/ANTs），在运行前先检查命令可用：
  - `command -v fslreorient2std` / `command -v flirt` 等
- 对于可重复执行的批处理：
  - 优先实现断点续跑（done/fail 记录，存在输出则跳过，除非 `--overwrite`）。

---

## 6) 变更与输出（写代码时）

- 修改代码使用最小改动原则，避免无关重构。
- 新增脚本：
  - 放到最贴近现有结构的目录（例如 `adapter_finetune/dataprocess/`）。
  - 自带 `--help` 参数说明、默认合理输出路径、日志路径与并行参数。
- 重要信息（输入路径、输出路径、关键假设）在运行前打印一次，便于排查。

---

## 7) 常用对话指令模板（直接复制到聊天输入框）

### 7.1 “用 xiaochou 跑一个脚本（带自动装包）”

> 请在 conda 环境 xiaochou 中运行：
> `python xxx.py --arg1 ...`。
> 如果缺少库，请自动安装后重试，并将日志写到 `logs_xxx/`。

### 7.2 “批处理 + 多进程 + 断点续跑”

> 请在 xiaochou 环境下对 `pairs.csv` 批处理，默认跳过已完成样本；失败写到 `fail.csv`，成功写到 `done.csv`；并行 `--jobs 10`。

### 7.3 “只要中文解释”

> 全程中文解释与输出；除非我要求英文。

---

## 8) 如何让它每次对话自动生效（推荐做法）

你可以选择以下任一方式，把本文件内容注入到每次 Copilot 对话中：

1. **VS Code 设置：Copilot Chat 自定义指令**
   - 打开设置（Settings）搜索：`Copilot Chat` / `Custom Instructions`（不同版本名称可能略有差异）。
   - 将本文件内容粘贴进去，或按设置项支持的方式引用该文件。

2. **工作区常驻文件（团队协作推荐）**
   - 将本文件保留在仓库根目录：`copilot-instructions.md`。
   - 团队成员可以统一引用/复制到各自的 Copilot 自定义指令配置中。

3. **快捷“开场指令”**
   - 若你的环境不支持自动加载文件，则把“语言+环境+自动安装”三条要求做成一段开场消息，开启新对话时第一条发送即可。
