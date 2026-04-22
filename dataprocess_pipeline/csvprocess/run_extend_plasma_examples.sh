#!/bin/bash
# 运行示例：使用不同的 MAX_DAYS 参数扩展 pairs_90d.csv

echo "================================================================"
echo "扩展 pairs_90d.csv 添加血浆信息 - 运行示例"
echo "================================================================"
echo ""

# 示例1: 使用默认的 90 天
echo "示例 1: MAX_DAYS=90 (默认)"
echo "----------------------------------------"
conda run -n xc python adapter_finetune/extend_pairs_with_plasma.py \
    --pairs /home/ssddata/linshuijin/replicaLT/adapter_finetune/gen_csv/pairs_180d_dx.csv \
    --upenn /home/ssddata/linshuijin/replicaLT/adapter_finetune/ADNI_csv/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv \
    --c2n /home/ssddata/linshuijin/replicaLT/adapter_finetune/ADNI_csv/C2N_PRECIVITYAD2_PLASMA_29Dec2025.csv \
    --output /home/ssddata/linshuijin/replicaLT/adapter_finetune/gen_csv/pairs_180d_with_plasma_90d.csv \
    --max-days 90

echo ""
echo ""

# 示例2: 使用更严格的 14 天
echo "示例 2: MAX_DAYS=14 (更严格的时间窗口)"
echo "----------------------------------------"
conda run -n xc python adapter_finetune/extend_pairs_with_plasma.py \
    --pairs /home/ssddata/linshuijin/replicaLT/adapter_finetune/gen_csv/pairs_180d_dx.csv \
    --upenn /home/ssddata/linshuijin/replicaLT/adapter_finetune/ADNI_csv/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv \
    --c2n /home/ssddata/linshuijin/replicaLT/adapter_finetune/ADNI_csv/C2N_PRECIVITYAD2_PLASMA_29Dec2025.csv \
    --output /home/ssddata/linshuijin/replicaLT/adapter_finetune/gen_csv/pairs_180d_with_plasma_14d.csv \
    --max-days 14

echo ""
echo ""

# 示例3: 使用更宽松的 60 天
echo "示例 3: MAX_DAYS=60 (更宽松的时间窗口)"
echo "----------------------------------------"
conda run -n xc python adapter_finetune/extend_pairs_with_plasma.py \
    --pairs /home/ssddata/linshuijin/replicaLT/adapter_finetune/gen_csv/pairs_180d_dx.csv \
    --upenn /home/ssddata/linshuijin/replicaLT/adapter_finetune/ADNI_csv/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv \
    --c2n /home/ssddata/linshuijin/replicaLT/adapter_finetune/ADNI_csv/C2N_PRECIVITYAD2_PLASMA_29Dec2025.csv \
    --output /home/ssddata/linshuijin/replicaLT/adapter_finetune/gen_csv/pairs_180d_with_plasma_60d.csv \
    --max-days 60

echo ""
echo ""
echo "================================================================"
echo "所有示例运行完成！"
echo "输出文件:"
echo "  - gen_csv/pairs_90d_with_plasma_90d.csv"
echo "  - gen_csv/pairs_90d_with_plasma_14d.csv"
echo "  - gen_csv/pairs_90d_with_plasma_60d.csv"
echo "================================================================"
