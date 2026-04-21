#!/bin/bash
# Monitor progress of the comparison pipeline
DIR="/mnt/nfsdata/nfsdata/lsj.14/replicaLT/analysis/comparison_results"

while true; do
    plasma_count=$(ls "$DIR/plasma/nifti/" 2>/dev/null | grep "_tau_pred" | wc -l)
    legacy_count=$(ls "$DIR/legacy/nifti/" 2>/dev/null | grep "_tau_pred" | wc -l)
    ficd_count=$(ls /mnt/nfsdata/nfsdata/lsj.14/replicaLT/runs/ficd_smoke_test/260415.3663353/predictions/eval/*.nii.gz 2>/dev/null | wc -l)
    
    echo "$(date '+%H:%M:%S') | Plasma: $plasma_count/43 | Legacy: $legacy_count/43 | FiCD: $ficd_count"
    
    # Check if main process is still running
    if ! ps aux | grep "run_comparison" | grep -v grep > /dev/null 2>&1; then
        echo "Main process finished!"
        break
    fi
    
    sleep 60
done
