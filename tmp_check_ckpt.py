import torch

ckpt_path = "adapter_v2/runs/checkpoints/best.pt"
state = torch.load(ckpt_path, map_location="cpu")

if "model_state_dict" in state:
    sd = state["model_state_dict"]
    print("[INFO] 使用 model_state_dict，顶层 keys:", list(state.keys()))
elif "state_dict" in state:
    sd = state["state_dict"]
    print("[INFO] 使用 state_dict，顶层 keys:", list(state.keys()))
else:
    sd = state
    print("[INFO] 直接使用顶层 dict（可能本身就是 state_dict）")

print(f"\n=== 全部 keys（共 {len(sd)} 个）===")
for k in sorted(sd.keys()):
    print(f"  {k:<60s}  {tuple(sd[k].shape)}")

print("\n=== 包含 'proj' 的 keys ===")
proj_keys = sorted(k for k in sd if "proj" in k.lower())
if proj_keys:
    for k in proj_keys:
        print(f"  {k:<60s}  {tuple(sd[k].shape)}")
else:
    print("  (无)")

print("\n=== 包含 'class' 的 keys ===")
cls_keys = sorted(k for k in sd if "class" in k.lower())
if cls_keys:
    for k in cls_keys:
        print(f"  {k:<60s}  {tuple(sd[k].shape)}")
else:
    print("  (无)")
