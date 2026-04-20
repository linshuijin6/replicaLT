#!/bin/bash
cd /home/data/linshuijin/replicaLT
nohup /home/data/linshuijin/.conda/envs/xiaochou/bin/python ficd_train.py --config configs/ficd/aligned_tau.yaml --gpu 3 > nohup_ficd_aligned_tau.log 2>&1 &
echo "STARTED PID: $!"
