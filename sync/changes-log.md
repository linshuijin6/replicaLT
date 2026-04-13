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
