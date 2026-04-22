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

## 2026-04-21 17:14 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `analysis/run_comparison.py` | 编辑 | 在`generate_unified_comparison()`中新增`_pasta_to_ref_space()`，修复PASTA(96,112,96)@1.5mm到参考空间的逆变换（净pad[(5,6),(8,8),(5,6)]→zoom），替换原错误的直接resize |

---
## 2026-04-21 22:02 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `analysis/run_comparison_pasta_exact.py` | 新建 | 新建 exact-match 对比脚本：PASTA 按 `name+examdate` 匹配，恢复到 `182x218x182@1mm` 后再中心裁到 `160x192x160`，unified 图中 PASTA diff 使用恢复后的 exact GT |
| `analysis/comparison_results_pasta_exact/pasta_exact_file_map.json` | 新建 | 保存 38 个 exact-match PASTA 结果的被试-时点映射 |
| `analysis/comparison_results_pasta_exact/pasta_missing_exact_subjects.json` | 新建 | 记录当前比较集里 5 个缺失 exact PASTA 时点的被试及可用日期 |
| `analysis/comparison_results_pasta_exact/pasta_gt_alignment_checks.csv` | 新建 | 保存恢复后的 PASTA GT 与 plasma GT 的对齐误差统计，平均 MAE 约 0.0430 |
| `analysis/comparison_results_pasta_exact/alignment_summary.md` | 新建 | 汇总 exact-match 覆盖情况、缺失时点清单和对齐统计 |
| `analysis/comparison_results_pasta_exact/figures/unified_comparison_exact_003_S_6260.png` | 新建 | 生成 representative unified 图，验证 PASTA 行不再出现整块错位深红 |
| `analysis/comparison_results_pasta_exact/figures/unified_comparison_exact_035_S_10241.png` | 新建 | 生成 representative unified 图，验证中位代表 subject 的 PASTA 对齐效果 |
| `analysis/comparison_results_pasta_exact/figures/unified_comparison_exact_941_S_6854.png` | 新建 | 生成 representative unified 图，验证末尾代表 subject 的 PASTA 对齐效果 |

---
## 2026-04-21 22:18 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `analysis/run_comparison_pasta_exact.py` | 编辑 | 将 PASTA 恢复链路改为“先回到 MNI，再复用 plasma 的 `CropForegroundd + ResizeWithPadOrCropd + NormalizeIntensityd + ScaleIntensityd` 投到公共显示空间”，修复 `941_S_6854` 绿圈区域黑白关系 |
| `analysis/comparison_results_pasta_exact/pasta_gt_alignment_checks.csv` | 编辑 | 用新显示链路重算 38 个 exact-match subject 的对齐统计，平均 MAE 从约 0.0430 降到 0.0218 |
| `analysis/comparison_results_pasta_exact/alignment_summary.md` | 编辑 | 更新 exact-match 摘要中的对齐统计数值 |
| `analysis/comparison_results_pasta_exact/figures/unified_comparison_exact_003_S_6260.png` | 编辑 | 按新的 plasma 公共显示空间重生成 representative unified 图 |
| `analysis/comparison_results_pasta_exact/figures/unified_comparison_exact_035_S_10241.png` | 编辑 | 按新的 plasma 公共显示空间重生成 representative unified 图 |
| `analysis/comparison_results_pasta_exact/figures/unified_comparison_exact_941_S_6854.png` | 编辑 | 按新的 plasma 公共显示空间重生成 representative unified 图，修复圈出区域的黑白反差 |

---

## 2026-04-21 22:37 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `analysis/run_comparison.py` | 编辑 | 修复 PASTA 空间对齐：新增 PLASMA_VAL_JSON 路径常量；重写 `_pasta_to_ref_space` 为三步流程（逆变换到 MNI → CropForeground → center crop/pad）；`generate_unified_comparison` 新增 raw_mri_paths 参数；加载原始 MRI 计算前景裁剪边界框 |

---

## 2026-04-21 22:56 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `analysis/run_comparison.py` | 编辑 | 移除PASTA可视化加载时的百分位缩放（除以GT p99.5），改为直接clip到[0,1]，修复unified图中PASTA偏亮问题 |

---

## 2026-04-22 20:25 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `README.md` | 编辑 | 全面更新以匹配代码实际状态：修正 plasma key 默认配置、TAU-only 训练权重、Stage B 模型架构/训练范式/验证指标、precompute_plasma_emb 信号流，删除过时章节并合并为术语表 |

---
