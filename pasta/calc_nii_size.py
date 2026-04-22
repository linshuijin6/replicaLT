import os
import json

DATA_DIR = "/mnt/nfsdata/nfsdata/lsj.14/PASTA/replicaLT_comparison/data"
TRAIN_JSON = os.path.join(DATA_DIR, "train_tabular.json")
VAL_JSON = os.path.join(DATA_DIR, "val_tabular.json")

def get_unique_files(json_path):
    files = set()
    if not os.path.exists(json_path):
        return files
    with open(json_path, 'r') as f:
        data = json.load(f)
        for item in data:
            if 'mri' in item and item['mri']:
                files.add(item['mri'])
            if 'tau' in item and item['tau']:
                files.add(item['tau'])
    return files

def main():
    train_files = get_unique_files(TRAIN_JSON)
    val_files = get_unique_files(VAL_JSON)
    all_files = train_files.union(val_files)
    
    total_size_bytes = 0
    missing_files = 0
    
    for f in all_files:
        if os.path.exists(f):
            total_size_bytes += os.path.getsize(f)
        else:
            missing_files += 1

    total_size_gb = total_size_bytes / (1024**3)
    print(f"Total unique NIfTI files needed: {len(all_files)}")
    print(f"Files found and sized: {len(all_files) - missing_files}")
    print(f"Missing files: {missing_files}")
    print(f"Total size: {total_size_gb:.2f} GB")

if __name__ == "__main__":
    main()
