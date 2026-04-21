# Changes Log

## 2026-04-12 14:38 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `plasma_train.py` | 编辑 | cache_dir 和 plasma_emb_dir 从 nfsdata 路径改为本地 NVMe 路径 /mnt/linshuijin/ |

---

## 2026-04-13 15:15 | 新建文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `plasma_inference.py` | 新建 | plasma_train.py checkpoint 的推理评估脚本，支持多步积分、NIfTI输出、三视面对比图、指标CSV/JSON |

---

## 2026-04-15 22:09 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `configs/ficd/aligned_tau.yaml` | 编辑 | pin_memory 改为 false，修复 torchio 兼容性 |
| `configs/ficd/strict_tau.yaml` | 编辑 | pin_memory 改为 false，修复 torchio 兼容性 |
| `ficd/config.py` | 编辑 | DEFAULT_CONFIG 中 pin_memory 默认值改为 False |
| `configs/ficd/smoke_test.yaml` | 新建 | 用于快速验证训练 loop 的 1-epoch 测试配置 |

---

## 2026-04-16 13:21 | 新建文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `AGENTS.md` | 新建 | 声明当前工作区遵循 `.agent/rules.md`，并在每次对话后触发 `sync-logger` |

---

## 2026-04-17 09:15 | 新建文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `ficd/ficd_review.ipynb` | 新建 | FICD 配置驱动的审查与可视化 notebook，27 个单元，复用 ficd 模块接口 |

---

## 2026-04-17 11:45 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `ficd/utils.py` | 编辑 | 新增兼容 checkpoint 状态字典提取逻辑，支持旧版 `model.pt` 和前缀清洗 |
| `ficd/ficd_review.ipynb` | 编辑 | 模型加载单元改为自动处理 `model.` 前缀与 shape mismatch 诊断 |

---

## 2026-04-17 12:11 | 新建文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `~/.agents/skills/network-debug/SKILL.md` | 新建 | 网络/代理诊断 skill，含 5 步决策树、6 个故障场景 |
| `~/.agents/skills/network-debug/scripts/diagnose.sh` | 新建 | 一键网络诊断脚本，检查隧道/环境变量/连通性/DNS/MCP |

---

## 2026-04-17 13:21 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `~/.agents/skills/network-debug/SKILL.md` | 编辑 | 新增 Auto-Trigger Rules、MCP 配置诊断章节（Notion OAuth/GitHub/TLS） |
| `~/.agents/skills/network-debug/scripts/diagnose.sh` | 编辑 | 修复 curl 返回码解析 bug |
| `~/.vscode-server/data/User/prompts/network-auto-detect.instructions.md` | 新建 | 描述匹配自动触发 instructions |
## 2026-04-21 14:10 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `ficd_train.py` | 编辑 | 训练loss加入x0_pred_loss（L1重建损失）；添加epoch_x0_pred_loss统计跟踪；日志输出加入x0_pred_loss |
| `configs/ficd/aligned_tau.yaml` | 编辑 | num_inference_steps从100改为1000，修复DDPM跳步推理导致的alpha计算错误 |

---

## 2026-04-21 16:10 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `analysis/run_comparison.py` | 编辑 | 重写 generate_unified_comparison()：布局改为6行×6列（3方向组×2行），GT统一用plasma文件夹，PASTA/FiCD预测逆向crop(pad)+scipy zoom重采样到GT shape |

---
