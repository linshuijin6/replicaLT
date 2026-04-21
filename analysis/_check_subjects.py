#!/usr/bin/env python3
"""Check subject overlap across all 4 methods."""
import json, os, re

# 1) val_json TAU subjects
with open('val_data_with_description.json') as f:
    val_data = json.load(f)
tau_entries = [e for e in val_data if e.get('tau') and '/zero/' not in e.get('tau','')]
tau_subjects = sorted(set(e['name'] for e in tau_entries))
print(f'val_json total: {len(val_data)}, with real TAU: {len(tau_entries)}, unique subjects: {len(tau_subjects)}')

# 2) fixed_split val_subjects
with open('fixed_split.json') as f:
    split = json.load(f)
val_subs = set(split['val_subjects'])
print(f'fixed_split val_subjects: {len(val_subs)}')

# 3) PASTA subjects
pasta_dir = '/mnt/nfsdata/nfsdata/lsj.14/PASTA/replicaLT_comparison/results/2026-04-12_331111/inference_output'
pasta_files = os.listdir(pasta_dir)
pasta_subjects = set()
pasta_file_map = {}
for fn in pasta_files:
    if fn.endswith('_syn_pet.nii.gz'):
        m = re.match(r'^(\d+_S_\d+)__', fn)
        if m:
            sid = m.group(1)
            pasta_subjects.add(sid)
            pasta_file_map[sid] = fn
print(f'PASTA subjects: {len(pasta_subjects)}')

# 4) Intersection
common = set(tau_subjects) & val_subs & pasta_subjects
print(f'\nCommon subjects (all 3): {len(common)}')
print('Subjects:', sorted(common))
missing_from_pasta = (set(tau_subjects) & val_subs) - pasta_subjects
print(f'\nIn val+tau but not in PASTA ({len(missing_from_pasta)}): {sorted(missing_from_pasta)}')

# 5) Check PASTA output shape
import nibabel as nib
first_pasta = next(iter(pasta_file_map.values()))
img = nib.load(os.path.join(pasta_dir, first_pasta))
print(f'\nPASTA output shape: {img.shape}, voxel size: {img.header.get_zooms()}')
print(f'Value range: [{img.get_fdata().min():.4f}, {img.get_fdata().max():.4f}]')

# Check a val_json TAU GT shape
sample = tau_entries[0]
if os.path.exists(sample['tau']):
    gt_img = nib.load(sample['tau'])
    print(f'TAU GT shape: {gt_img.shape}, voxel size: {gt_img.header.get_zooms()}')
if os.path.exists(sample['mri']):
    mri_img = nib.load(sample['mri'])
    print(f'MRI shape: {mri_img.shape}, voxel size: {mri_img.header.get_zooms()}')

# Save common subjects list for later use
with open('analysis/_common_subjects.json', 'w') as f:
    json.dump(sorted(common), f, indent=2)
print(f'\nSaved {len(common)} common subjects to analysis/_common_subjects.json')
