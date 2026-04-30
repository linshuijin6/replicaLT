# Changes Log

## 2026-04-30 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `plasma_train.py` | 编辑 | 三 Token 并列方案：新增 Section 4.5 预计算 clinical_emb，index_transform 从 (2,512) 扩展为 (3,512)，cache_dir 改为 `_3tok` 避免旧缓存污染 |

---

## 2026-04-24 14:20 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `analysis/run_comparison.py` | 编辑 | 修复 `_common_subjects.json` 改为 `{name, examdate}` 列表后触发的 `load_subjects()` 字典排序报错；兼容新旧 subject 格式；PASTA 文件匹配改为结合 `val_43_subjects.json` 的 `examdate/diagnosis` 做精确筛选；FiCD 结果目录改为自动解析 `run_dir`/`inference`/`predictions` 布局 |
| `analysis/comparison_results/comparison_report.md` | 编辑 | 重新运行 `analysis/run_comparison.py` 后更新比较报告 |
| `sync/chat-log.md` | 编辑 | 记录本次 `run_comparison.py` 报错修复与验证结果 |
| `sync/changes-log.md` | 编辑 | 追加本次代码与输出更新日志 |

---

## 2026-04-12 14:38 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `plasma_train.py` | 编辑 | cache_dir 和 plasma_emb_dir 从 nfsdata 路径改为本地 NVMe 路径 /mnt/linshuijin/ |

---

## 2026-04-23 10:39 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `analysis/run_comparison.py` | 编辑 | 重构为配置驱动运行脚本：新增四方法嵌套配置、`RUN_MODE`/`ENABLED_METHODS`、PASTA 推理入口、统一 `OUT_DIR` 输出、fail-fast 校验、按启用方法动态统计与绘图 |
| `sync/chat-log.md` | 编辑 | 记录本次 `run_comparison.py` 运行逻辑重构摘要 |
| `sync/changes-log.md` | 编辑 | 追加本次代码改动日志 |

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

## 2026-04-16 10:42 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `configs/ficd/aligned_tau.yaml` | 编辑 | 将 num_inference_steps 从 1000 改为 100，解决val极慢问题 |
| `configs/ficd/strict_tau.yaml` | 编辑 | 将 num_inference_steps 从 1000 改为 100，解决val极慢问题 |
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

## 2026-04-22 09:56 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `ficd_train.py` | 编辑 | 删除额外的 compute_x0_predictions 辅助函数，将 train 中的 x0 恢复逻辑内联为 notebook 同款 scheduler.step 循环，并为 validate 的 inferer.sample 补充 autocast 混合精度 |

---

## 2026-04-22 10:20 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `configs/ficd/aligned_tau.yaml` | 编辑 | 将网络配置改为与 FICD notebook 一致，开启 with_conditioning 并补充 cross_attention_dim=64 |

---

## 2026-04-22 10:52 | 新建文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `analysis/benchmark_ficd_run_validation.py` | 新建 | 新增 run_validation 随机数据基准脚本，验证当前验证路径可运行性并实测 48 个 3D 样本的推理耗时，支持 checkpoint 兼容加载与 1000 步外推估算 |
## 2026-04-22 20:25 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `README.md` | 编辑 | 全面更新以匹配代码实际状态：修正 plasma key 默认配置、TAU-only 训练权重、Stage B 模型架构/训练范式/验证指标、precompute_plasma_emb 信号流，删除过时章节并合并为术语表 |

---

## 2026-04-22 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `ficd/config.py` | 编辑 | `load_config` 中 `repo_root = config_path.parents[2]` 改为 `parents[1]`，修复 JSON 路径解析错误 |

---

## 2026-04-23 10:03 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `analysis/run_comparison.py` | 编辑 | 新增 FIG_DIR 到头部路径块，将 main() 内三处硬编码路径替换为头部变量 FIG_DIR 和 COMMON_SUBJECTS |

---

## 2026-04-23 18:05 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `analysis/extract_test_data_with_description.py` | 新建 | 新增最小提取脚本，按 `analysis/_common_subjects.json` 中的 subject 过滤 `val_data_with_description.json`，输出 `test_data_with_description.json` |
| `test_data_with_description.json` | 新建 | 从 `val_data_with_description.json` 抽取出 56 条记录，覆盖 43 个 common subjects |
| `sync/chat-log.md` | 编辑 | 记录本次 test split 提取任务摘要 |
| `sync/changes-log.md` | 编辑 | 追加本次代码与数据文件变更日志 |

---

## 2026-04-23 18:18 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `analysis/_check_subjects.py` | 编辑 | 将样本校验从仅按 subject 改为按 `(name, examdate)` 唯一匹配；移除 `fixed_split.json` 和 PASTA 输出文件名依赖；四方法 JSON 路径改为硬编码 |
| `analysis/_common_subjects.json` | 编辑 | 输出格式从 subject 字符串列表改为 `{name, examdate}` 对列表，保存四方法公共样本 88 对 |
| `sync/chat-log.md` | 编辑 | 记录本次 `_check_subjects.py` 校验逻辑修正摘要 |
| `sync/changes-log.md` | 编辑 | 追加本次样本匹配逻辑和输出格式变更日志 |

---

## 2026-04-24 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `analysis/run_comparison.py` | 编辑 | 修复 FiCD unified 可视化：删除二次 `(raw + 1) / 2`，新增 `[0,1]` 值域断言、FiCD `hparams.json` crop/target_shape 读取、按 `examdate` 和 MRI/TAU 路径一致性校验，以及 `160×180×160 -> 182×218×182 -> 160×192×160` 的显示空间还原 |
| `analysis/comparison_results/comparison_report.md` | 生成/编辑 | 执行 `run_comparison.py` 后重新生成比较报告 |
| `analysis/comparison_results/figures/unified_comparison_006_S_6441.png` | 生成/编辑 | 使用修复后的 FiCD 对齐逻辑生成指定 subject 的 unified 图，用于确认背景和尺寸问题已修复 |
| `analysis/comparison_results/figures/unified_comparison_114_S_6429.png` | 生成/编辑 | 执行 `run_comparison.py` 后自动刷新 unified 图 |
| `analysis/comparison_results/figures/unified_comparison_130_S_6688.png` | 生成/编辑 | 执行 `run_comparison.py` 后自动刷新 unified 图 |
| `analysis/comparison_results/figures/unified_comparison_033_S_6889.png` | 生成/编辑 | 执行 `run_comparison.py` 后自动刷新 unified 图 |

---

## 2026-04-29 20:54 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `~/.codex/AGENTS.md` | 编辑 | 写入 Codex 用户级 Sync Logger 完成门槛，使每轮对话自动记录 chat-log，并在文件变更时记录 changes-log 与尝试 Notion 同步 |
| `~/.config/sync-logger.instructions.md` | 新建 | 新增用户级同步规则源副本，保存 Codex 自动日志与 Notion 同步流程 |

---
## 2026-04-29 22:24 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `validation/E1/compute_real_av45_suvr.py` | 新建 | 新增真实 AV45/FBP composite SUVR 计算脚本，按 Lan 2025 的 frontal/cingulate/parietal/temporal 目标 ROI 和 whole cerebellum 参考区定义 Aβ+/- 并生成 ROC/AUC |
| `validation/E1/README.md` | 新建 | 记录 E1 验证任务命令、方法、4D PET 处理方式和 ROC/AUC 解释约束 |
| `validation/E1/results/real_av45_suvr.csv` | 新建 | 保存 388 条成功计算的真实 AV45 SUVR、Aβ 标签和四类 ROI 分项均值 |
| `validation/E1/results/failed_samples.csv` | 新建 | 保存 2 条全零 PET 失败样本及错误原因 |
| `validation/E1/results/roi_labels_used.json` | 新建 | 保存 Harvard-Oxford 目标 ROI 标签、AAL3 小脑标签和 atlas 缓存来源 |
| `validation/E1/results/roc_auc.json` | 新建 | 保存阈值 1.11、Aβ-/Aβ+ 数量 185/203 和 ROC/AUC sanity check |
| `validation/E1/results/roc_curve.csv` | 新建 | 保存 ROC 曲线点 |
| `validation/E1/results/roc_curve.png` | 新建 | 保存真实 AV45 SUVR ROC 图 |
| `validation/E1/results/suvr_histogram.png` | 新建 | 保存真实 AV45 composite SUVR 分布直方图 |
| `validation/E1/results/summary.json` | 新建 | 保存完整运行摘要，记录 390 条真实 AV45、388 条成功、2 条失败 |
| `validation/E1/nilearn_data/` | 新建 | 固定缓存 nilearn Harvard-Oxford atlas 和 AAL3 fallback 数据，保证脚本复跑可复用 |

---

## 2026-04-30 10:43 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `~/.config/sync-logger.instructions.md` | 编辑 | 用完整详细版规则（含 YAML frontmatter、规则 1-5、防跳过机制）替换旧的简化版 |

---
## 2026-04-30 10:49 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `plasma_train_codex.py` | 新建 | 新增 3-token 训练入口，将 `old_descr`/`description` 经 BiomedCLIP 编码为 `clinical_emb`，组成 `[plasma_emb, clinical_emb, modality_text]` context，并使用独立 3tok 缓存目录 |
| `.agent/rules.md` | 编辑 | 补充 `plasma_train_codex.py` 作为三 token 条件变体入口及对应 context shape、BiomedCLIP 用途和检索优先级 |
| `.github/copilot-instructions.md` | 编辑 | 同步记录三 token 条件变体入口、token 含义和文档检查范围 |

---

## 2026-04-30 15:07 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `adapter_v2/config.yaml` | 编辑 | 新增 `data.tracer`、`model.ctx_init`，更新 `prompt_template` 为 AV45 |
| `adapter_v2/dataset.py` | 编辑 | `TAUPlasmaDataset` 增加 `pet_id_column` 参数，替换 3 处 `id_av1451` 硬编码，`tau_id` key 改为 `pet_id` |
| `adapter_v2/models.py` | 编辑 | `CoCoOpTAUModel` 增加 `ctx_init` 参数，替换内部硬编码初始化 token |
| `adapter_v2/train.py` | 编辑 | 解析 tracer 字段，传参到 `get_cache_stats`/`generate_missing_caches`/`TAUPlasmaDataset`/`CoCoOpTAUModel`（共 8 处） |

---

## 2026-04-30 15:30 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `adapter_v2/train.py` | 编辑 | `_TRACER_MAP` 加入 `prompt_template_default`，两者改为 tracer 派生+可选 override |
| `adapter_v2/config.yaml` | 编辑 | `prompt_template` 和 `ctx_init` 改为注释状态，由 `data.tracer` 唯一控制 |

---

## 2026-04-30 16:04 | 多文件修改

| 文件 | 操作 | 摘要 |
|------|------|------|
| `adapter_v2/precompute_cache.py` | 编辑 | build_sample_list 支持 av45/fdg modality（已于上轮完成） |
| `adapter_v2/retrieval_eval.py` | 编辑 | 加 _TRACER_MAP，PlasmaTextEncoder/ClassTextEncoder/DiagnosisTextRetrievalEvaluator 新增 tracer 参数 |
| `adapter_v2/run_probe_and_retrieval.py` | 编辑 | parse_args 加 --tracer；ContextConditionedClassifier 加 tracer，透传 ctx_init/class_prompt_template |

---
