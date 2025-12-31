from torch.utils.data import Dataset, DataLoader
import h5py
import torch
from tqdm import tqdm
import torch
import os
import numpy as np
import time

def save_to_hdf5(data_list, transform, h5_path):
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        for i, item in enumerate(tqdm(data_list)):
            processed = transform(item)  # transform 后的字典
            grp = f.create_group(f"sample_{i}")
            for k, v in processed.items():
                if isinstance(v, torch.Tensor):
                    grp.create_dataset(k, data=v.cpu().numpy(), compression="gzip")
                elif isinstance(v, np.ndarray):
                    grp.create_dataset(k, data=v, compression="gzip")
                elif isinstance(v, (int, float)):
                    grp.create_dataset(k, data=v)
                elif isinstance(v, str):
                    grp.create_dataset(k, data=np.string_(v))  # str -> bytes
                else:
                    # 其他类型可以保存为字符串
                    grp.create_dataset(k, data=str(v))

    print(f"Saved {len(data_list)} samples to {h5_path}")


class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # 打开文件并获取样本数量
        with h5py.File(self.h5_path, "r") as f:
            self.keys = list(f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            grp = f[self.keys[idx]]
            item = {}
            for k, v in grp.items():
                if isinstance(v, h5py.Dataset):
                    val = v[:]
                    if val.dtype.kind == "S":  # bytes -> str
                        val = val.astype(str).tolist()
                        if len(val) == 1:
                            val = val[0]
                    # 转为 tensor
                    if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
                        val = torch.tensor(val)
                    item[k] = val

        return item

def benchmark_dataloader(dataset, batch_size=1, num_workers_list=[0,2,4,8]):
    for nw in num_workers_list:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=nw)
        start = time.time()
        for i, data in enumerate(loader):
            if i >= 10:  # 只跑前 10 个 batch
                break
        end = time.time()
        print(f"num_workers={nw}, time for 10 batches: {end-start:.2f}s")


if __name__ == "__main__":
    # 创建 DataLoader
    train_ds = HDF5Dataset("./processed_data/train.h5")
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

    val_ds = HDF5Dataset("./processed_data/val.h5")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
