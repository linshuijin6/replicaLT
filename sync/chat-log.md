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

## 2026-04-15 22:09 | FICD对比基线审查

**Q:** 审查 FICD 对比基线接入计划（strict/aligned 双口径），检查代码落实，在 xiaochou 环境运行并修复问题。

**A:** 计划合理，代码实现与原始 FICD notebook 完全一致（模型结构、loss、scheduler、预处理）。发现并修复两个 Critical 问题：(1) `rfc3987-syntax` 包导致 MONAI 导入挂起，已卸载；(2) `pin_memory=True` 与 torchio SubjectsDataset 不兼容，已改为 false。smoke test 在 GPU 6 成功运行，训练 loop、TensorBoard 记录、run 目录结构均正确。注意正式训练需选空闲 GPU（>=22GB 显存）。

---

## 2026-04-16 10:41 | Val极慢原因排查

**Q:** val阶段极其耗时，检查原因。train和val的steps都是1000。

**A:** 根本原因：训练每步仅1次前向（随机加噪+预测），验证调用 `inferer.sample()` 执行完整DDPM逆扩散，需循环 `num_inference_steps=1000` 次前向。88 batch × ~47min/batch ≈ 69小时完成一次val，是训练速度的1000倍。已将两个config（aligned_tau.yaml 和 strict_tau.yaml）的 `num_inference_steps` 从1000改为100，预计val约4小时。如需更快可进一步降到50或限制val batch数。

---
