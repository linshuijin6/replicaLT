"""
生成数据集缓存的独立脚本。
运行命令:
python baseline/precompute_cache.py
"""
import sys
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# 添加父目录到环境变量，以便按包结构导入 baseline 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.dataset import create_dataloaders
from baseline.config import get_default_config

def main():
    parser = argparse.ArgumentParser(description="Precompute NIfTI Cache for Faster Training")
    # 默认设为 NFS 以避免 SSD 满了，如果你清理了 SSD，可以自行改为 /mnt/ssd/linshuijin/adni_nifti_cache
    parser.add_argument("--cache_dir", type=str, default="/mnt/nfsdata/nfsdata/lsj.14/ADNI_base_cache", help="预计算的缓存目录")
    parser.add_argument("--target_pet", type=str, choices=["tau", "fdg", "av45"], default="tau", help="目标 PET 模态 (tau/fdg/av45)")
    parser.add_argument("--num_workers", type=int, default=8, help="多进程并发生成缓存的进程数")
    args = parser.parse_args()

    # 强制将工作目录切换到项目根目录（例如 replicaLT），避免因 CWD 不同导致相对路径（如 adapter_finetune/...csv）找不到
    project_root = str(Path(__file__).resolve().parent.parent)
    os.chdir(project_root)

    config = get_default_config()
    
    # 强制覆盖配置以符合缓存生成需求
    config.data.use_cache = True
    config.data.cache_dir = args.cache_dir
    config.data.target_pet = args.target_pet
    config.data.num_workers = args.num_workers
    config.data.persistent_workers = False # 这里只跑一次，没必要持久化
    
    print("=" * 60)
    print(f"🚀 开始为 {args.target_pet.upper()} 生成数据缓存")
    print(f"📦 缓存目标目录: {args.cache_dir}")
    print(f"⚙️  并发写入进程数: {args.num_workers}")
    print("=" * 60)
    
    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)

    print("\n1. 解析数据列表并初始化 Dataset (不训练数据)...")
    train_loader, val_loader, test_loader, train_df, val_df, test_df = create_dataloaders(config)

    # 开启 cache_only 标志：当文件已存在于缓存中时，DataLoader 底层立刻跳出，杜绝 19MB/文件的 NFS 文件 I/O 加载耗时
    train_loader.dataset.cache_only = True
    val_loader.dataset.cache_only = True
    test_loader.dataset.cache_only = True

    # 重新包装 DataLoader：
    # 1. 设置 batch_size=8 合理分配给进程
    # 2. collate_fn=lambda x: None 是为了在读并写完缓存后，跳过在内存中把大数组合并为 Tensor 的高开销操作，释放并节省大量内存
    train_cache_loader = DataLoader(
        train_loader.dataset, batch_size=8, num_workers=args.num_workers, shuffle=False, collate_fn=lambda x: None
    )
    val_cache_loader = DataLoader(
        val_loader.dataset, batch_size=8, num_workers=args.num_workers, shuffle=False, collate_fn=lambda x: None
    )
    test_cache_loader = DataLoader(
        test_loader.dataset, batch_size=8, num_workers=args.num_workers, shuffle=False, collate_fn=lambda x: None
    )

    loaders = [
        ("Train Set", train_cache_loader, len(train_df)),
        ("Val Set", val_cache_loader, len(val_df)),
        ("Test Set", test_cache_loader, len(test_df))
    ]
    
    print("\n2. 开始极速并行读取与计算（内部自动生成 .npy 文件）...")
    for segment_name, loader, total_samples in loaders:
        print(f"\n>> {segment_name} (处理数: {total_samples})...")
        # 调用 Dataset 的同时自动激活 _load_and_preprocess 写缓存机制
        for _ in tqdm(loader, desc=segment_name, total=len(loader)):
            pass

    print("\n✅ 所有数据缓存生成完毕，预计算完成！")
    print("=" * 60)
    print(f"接下来，你可以在 run_experiments.sh 等正式训练环节中，直接传入：")
    print(f"   --cache_dir {args.cache_dir}")
    print(f"或者任由 config 读取自动设定它。训练将立即拉起缓存好的 .npy，极速打满 GPU。")

if __name__ == "__main__":
    main()
