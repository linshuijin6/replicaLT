from time import sleep

import torch

def find_max_allocation(device="cuda:1"):
    GB = 1024 ** 3
    alloc = 18
    while True:
        try:
            x = torch.empty(int(alloc * GB / 4), device=device, dtype=torch.float32)
            x.fill_(1.0)
            print(f"Allocated {alloc} GB successfully")
            alloc += 1
            sleep(1000000)
        except RuntimeError:
            print(f"❌ Failed at {alloc} GB")
            break

find_max_allocation()