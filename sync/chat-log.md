# Chat Log

## 2026-04-12 14:24 | 缓存迁移可行性分析

**Q:** 检查 plasma_train.py 使用的缓存目录，分析其大小，评估迁移到非 nfsdata 路径的可行性，特别关注本地磁盘剩余空间。

**A:** plasma_train.py 使用两个缓存：PersistentDataset 缓存（ADNI_cache0312，train 949 文件 ~69.5GB + val 95 文件 ~7.0GB，共约 76.5GB）和 plasma_emb 缓存（ADNI_plasma_cache，906 文件仅 ~3MB）。本地磁盘中 /mnt/ssd 仅剩 9.9G 不可行；NVMe 系统盘（/dev/nvme0n1p2）剩余 158G 可容纳全部缓存，建议迁移到 /home/linshuijin/ 下。

---

## 2026-04-12 14:33 | 缓存迁移到本地NVMe

**Q:** 执行完整迁移（复制方式），完成后修改代码路径。

**A:** 目标路径选定 /mnt/linshuijin/（NVMe 系统盘，用户拥有）。plasma_cache（906 文件 ~3MB）已复制完成。ADNI_cache0312（~76.5GB）通过 rsync 后台复制中（PID 3674435，NFS 速率 ~1.73MB/s）。plasma_train.py 中 cache_dir 和 plasma_emb_dir 已改为本地路径。原始 nfsdata 副本保留不动。

---

## 2026-04-13 15:15 | plasma推理脚本

**Q:** 为 plasma_train.py 训练好的 checkpoint 撰写测试/推理代码，需先设计计划再实现。

**A:** 创建了 plasma_inference.py（530行），核心设计：Token 0 使用预计算 plasma_emb、Token 1 使用 BiomedCLIP TAU 模态文本（与训练完全一致）；支持 rectified-flow Euler 多步积分（--n_steps）；自动适配单/多 GPU 权重；逐样本计算 SSIM/PSNR/MAE/MSE；输出 NIfTI、三视面对比图 PNG、metrics.csv、summary.json。用法：`python3 plasma_inference.py --ckpt runs/xxx/best_model.pt --gpu 0`。

---
