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

## 2026-04-16 10:42 | 编辑文件

| 文件 | 操作 | 摘要 |
|------|------|------|
| `configs/ficd/aligned_tau.yaml` | 编辑 | 将 num_inference_steps 从 1000 改为 100，解决val极慢问题 |
| `configs/ficd/strict_tau.yaml` | 编辑 | 将 num_inference_steps 从 1000 改为 100，解决val极慢问题 |

---
