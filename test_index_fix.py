#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试索引修复是否正确"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import torch
from transformers import AutoProcessor, AutoModel

# 加载模型
local_model_path = './BiomedCLIP'
processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)
device = torch.device('cuda:0')
model.to(device)
model.eval()

# 加载数据
with open('./train_data_with_description.json', 'r') as f:
    train_data = json.load(f)
with open('./val_data_with_description.json', 'r') as f:
    val_data = json.load(f)

print(f'train_data: {len(train_data)}')
print(f'val_data: {len(val_data)}')

# 检查原始JSON中的索引值
print('\n原始JSON中的索引示例:')
if len(train_data) > 0:
    print(f'  train_data[0]: tau_index={train_data[0].get("tau_index")}, av45_index={train_data[0].get("av45_index")}')
if len(val_data) > 0:
    print(f'  val_data[0]: tau_index={val_data[0].get("tau_index")}, av45_index={val_data[0].get("av45_index")}')
    print(f'  val_data[-1]: tau_index={val_data[-1].get("tau_index")}, av45_index={val_data[-1].get("av45_index")}')

# 重新分配索引（修复后的方式）
train_data_fixed = [
    {'description': item.get('old_descr') or '', 'tau_index': idx, 'av45_index': idx}
    for idx, item in enumerate(train_data)
]
val_data_fixed = [
    {'description': item.get('old_descr') or '', 'tau_index': idx, 'av45_index': idx}
    for idx, item in enumerate(val_data)
]

print('\n修复后的索引示例:')
if len(train_data_fixed) > 0:
    print(f'  train_data_fixed[0]: tau_index={train_data_fixed[0]["tau_index"]}, av45_index={train_data_fixed[0]["av45_index"]}')
if len(val_data_fixed) > 0:
    print(f'  val_data_fixed[0]: tau_index={val_data_fixed[0]["tau_index"]}, av45_index={val_data_fixed[0]["av45_index"]}')
    print(f'  val_data_fixed[-1]: tau_index={val_data_fixed[-1]["tau_index"]}, av45_index={val_data_fixed[-1]["av45_index"]}')

# 构建 paired_data
paired_data = train_data_fixed + val_data_fixed
modal_information = [data['description'] for data in paired_data if data['description'] is not None]
print(f'\nmodal_information: {len(modal_information)}')

# 编码文本
text_inputs = processor(modal_information, padding=True, truncation=True, return_tensors='pt', max_length=256).to(device)
with torch.no_grad():
    desc_text_features = model.get_text_features(text_inputs['input_ids'])
desc_text_features_cpu = desc_text_features.cpu()
print(f'desc_text_features_cpu: {desc_text_features_cpu.shape}')

# 测试索引
print(f'\n验证索引范围:')
print(f'  train_data 索引范围: 0 ~ {len(train_data_fixed)-1}')
print(f'  val_data 索引范围: 0 ~ {len(val_data_fixed)-1}')
print(f'  val_data 对应 desc_text_features 索引: {len(train_data_fixed)} ~ {len(train_data_fixed)+len(val_data_fixed)-1}')
print(f'  desc_text_features_cpu 大小: {desc_text_features_cpu.shape[0]}')

# 测试最大索引
max_val_idx = len(val_data_fixed) - 1
actual_idx = max_val_idx + len(train_data_fixed)
print(f'\n测试最大索引:')
print(f'  val_data 最大本地索引: {max_val_idx}')
print(f'  对应 desc_text_features 索引: {actual_idx}')
print(f'  是否越界: {actual_idx >= desc_text_features_cpu.shape[0]}')

if actual_idx < desc_text_features_cpu.shape[0]:
    print('\n✅ 索引修复成功！')
else:
    print('\n❌ 仍然越界，需要进一步检查')
