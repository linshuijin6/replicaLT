import json

# 读取的txt文件路径
input_file = "/home/ssddata/liutuo/liutuo_data/naoqu_name.txt"
# 输出的json文件路径
output_file = "./brain_labels.json"

data = {}

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # 跳过空行和注释行
        if not line or line.startswith("#"):
            continue
        
        # 按空白字符分割
        parts = line.split()
        
        # 确保至少有两部分（编号 + 名称）
        if len(parts) < 2:
            continue
        
        # 第一个是编号
        num = parts[0]
        
        # 名称可能中间有连接符，因此要拼接除编号和最后4个RGB数字外的部分
        # 若行中末尾是R G B A四个数，则去掉最后4个
        if len(parts) > 5:
            name = " ".join(parts[1:-4])
        else:
            name = parts[1]
        
        data[num] = name

# 写入JSON文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"✅ 已成功生成 {output_file}")
