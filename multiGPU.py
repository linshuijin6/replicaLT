import torch
import torch.nn as nn
from generative.networks.nets.diffusion_model_unet import DistributedDiffusionModelUNet
# 定义模型
model = DistributedDiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,  # 1,
    out_channels=1,
    num_channels=(2,2,2,2),
    attention_levels=(False, False, False, True),
    num_res_blocks=1,
    num_head_channels=(0, 0, 0, 2),
    with_conditioning=True,
    cross_attention_dim=2,
    use_flash_attention=False,
    norm_num_groups=1,
    device_ids=[0,2])

# 输入数据
x = torch.randn(1, 1, 8, 8, 8).to('cuda:0')
time = torch.randint(0, 1000, (1,)).to('cuda:0')
context = torch.randn(1, 2, 2).to('cuda:0')
out = model(x=x, timesteps=time, context=context)

loss = out.sum()
loss.backward()
