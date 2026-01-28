"""
生成包含人口统计学信息的CSV表格
字段：subject_id, examdate, sex, weight, diagnosis, age, description
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
import os

def parse_date(date_str):
    """解析日期字符串，支持多种格式"""
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        # 尝试多种日期格式
        for fmt in ['%Y/%m/%d', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S']:
            try:
                return datetime.strptime(str(date_str).split()[0], fmt)
            except:
                continue
        return None
    except:
        return None

def calculate_examdate_from_visit(entry_date, visit):
    """
    根据entry_date和visit计算examdate
    - visit为bl时，examdate = entry_date
    - visit为m*时，examdate = entry_date + *个月
    - visit为sc时，examdate = entry_date (screening visit)
    """
    if entry_date is None:
        return None
    
    if pd.isna(visit) or visit == '':
        return entry_date
    
    visit = str(visit).lower().strip()
    
    if visit in ['bl', 'sc', 'scmri']:
        return entry_date
    
    # 解析 m* 格式，如 m06, m12, m24 等
    match = re.match(r'm(\d+)', visit)
    if match:
        months = int(match.group(1))
        return entry_date + relativedelta(months=months)
    
    return entry_date

def get_birth_year(df_subject, subject_id):
    """获取subject的出生年份（PTDOBYY）"""
    subject_rows = df_subject[df_subject['subject_id'] == subject_id]
    for _, row in subject_rows.iterrows():
        ptdobyy = row.get('PTDOBYY')
        if pd.notna(ptdobyy) and ptdobyy != '':
            try:
                # PTDOBYY 可能是 "1950-01-01" 格式或单独年份
                if isinstance(ptdobyy, str) and '-' in ptdobyy:
                    return int(ptdobyy.split('-')[0])
                else:
                    return int(float(ptdobyy))
            except:
                continue
    return None

def find_closest_row(df_mytable_subject, target_examdate):
    """
    在同一subject的my_table行中找到examdate最接近的行
    """
    if df_mytable_subject.empty or target_examdate is None:
        return None
    
    min_diff = float('inf')
    closest_row = None
    
    for idx, row in df_mytable_subject.iterrows():
        row_examdate = row.get('calculated_examdate')
        if row_examdate is None:
            continue
        
        diff = abs((row_examdate - target_examdate).days)
        if diff < min_diff:
            min_diff = diff
            closest_row = row
    
    return closest_row

def convert_weight(vsweight, vswtunit):
    """
    转换体重为kg
    - VSWTUNIT=1: pounds，需要转换为kg (1 pound = 0.453592 kg)
    - VSWTUNIT=2: kg，直接使用
    """
    if pd.isna(vsweight) or vsweight == '':
        return None
    
    try:
        weight = float(vsweight)
    except:
        return None
    
    if pd.isna(vswtunit) or vswtunit == '':
        # 如果单位未知，根据数值大小推断
        # 一般体重超过100的可能是pounds
        if weight > 100:
            return round(weight * 0.453592, 1)
        else:
            return round(weight, 1)
    
    try:
        unit = int(float(vswtunit))
    except:
        return round(weight, 1)
    
    if unit == 1:  # pounds
        return round(weight * 0.453592, 1)
    elif unit == 2:  # kg
        return round(weight, 1)
    else:
        return round(weight, 1)

def convert_sex(ptgender):
    """转换性别：1=male, 2=female"""
    if pd.isna(ptgender) or ptgender == '':
        return None
    
    try:
        gender = int(float(ptgender))
        if gender == 1:
            return 'male'
        elif gender == 2:
            return 'female'
    except:
        pass
    return None

def convert_diagnosis(diagnosis):
    """转换诊断：1=CN, 2=MCI, 3=AD"""
    if pd.isna(diagnosis) or diagnosis == '':
        return None
    
    try:
        diag = int(float(diagnosis))
        if diag == 1:
            return 'CN'
        elif diag == 2:
            return 'MCI'
        elif diag == 3:
            return 'AD'
    except:
        pass
    return None

def generate_description(age, sex, weight):
    """生成description文本"""
    if age is None or sex is None or weight is None:
        # 部分信息缺失时，尽可能生成描述
        parts = []
        if age is not None:
            parts.append(f"{age}-year-old")
        if sex is not None:
            parts.append(sex)
        if weight is not None:
            parts.append(f"with a weight of {weight} kg")
        
        if not parts:
            return None
        
        if age is not None and sex is not None and weight is not None:
            return f"Subject is a {age}-year-old {sex} with a weight of {weight} kg."
        elif age is not None and sex is not None:
            return f"Subject is a {age}-year-old {sex}."
        elif age is not None:
            return f"Subject is {age} years old."
        else:
            return None
    
    return f"Subject is a {age}-year-old {sex} with a weight of {weight} kg."

def main():
    # 设置路径
    base_path = '/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/data_csv'
    pairs_path = os.path.join(base_path, 'pairs_withPlasma.csv')
    mytable_path = os.path.join(base_path, 'All_Subjects_My_Table_20Jan2026.csv')
    output_path = os.path.join(base_path, 'pairs_with_demog.csv')
    
    print("读取数据文件...")
    # 读取pairs表格
    df_pairs = pd.read_csv(pairs_path)
    print(f"pairs表格: {len(df_pairs)} 行")
    
    # 读取my_table表格
    df_mytable = pd.read_csv(mytable_path)
    print(f"my_table表格: {len(df_mytable)} 行")
    
    # 1. 计算my_table中每行的examdate
    print("\n计算my_table中每行的examdate...")
    df_mytable['entry_date_parsed'] = df_mytable['entry_date'].apply(parse_date)
    df_mytable['calculated_examdate'] = df_mytable.apply(
        lambda row: calculate_examdate_from_visit(row['entry_date_parsed'], row['visit']),
        axis=1
    )
    
    # 预先为每个subject计算出生年份
    print("获取出生年份...")
    subject_birth_years = {}
    for subject_id in df_mytable['subject_id'].unique():
        birth_year = get_birth_year(df_mytable, subject_id)
        if birth_year:
            subject_birth_years[subject_id] = birth_year
    print(f"找到 {len(subject_birth_years)} 个subject的出生年份")
    
    # 按subject_id分组my_table
    mytable_grouped = df_mytable.groupby('subject_id')
    
    # 2. 为pairs表格中的每一行找到对应的my_table行
    print("\n匹配pairs和my_table...")
    results = []
    matched_count = 0
    
    for idx, pair_row in df_pairs.iterrows():
        subject_id = pair_row['PTID']
        examdate_str = pair_row['EXAMDATE']
        target_examdate = parse_date(examdate_str)
        
        # 初始化结果
        result = {
            'subject_id': subject_id,
            'examdate': examdate_str,
            'sex': None,
            'weight': None,
            'diagnosis': None,
            'age': None,
            'description': None
        }
        
        # 查找匹配的my_table行
        if subject_id in mytable_grouped.groups:
            df_subject = mytable_grouped.get_group(subject_id)
            closest_row = find_closest_row(df_subject, target_examdate)
            
            if closest_row is not None:
                matched_count += 1
                
                # 3. 获取性别
                result['sex'] = convert_sex(closest_row.get('PTGENDER'))
                
                # 4. 获取体重
                result['weight'] = convert_weight(
                    closest_row.get('VSWEIGHT'),
                    closest_row.get('VSWTUNIT')
                )
                
                # 5. 获取诊断
                result['diagnosis'] = convert_diagnosis(closest_row.get('DIAGNOSIS'))
                
                # 计算年龄
                if subject_id in subject_birth_years and target_examdate is not None:
                    birth_year = subject_birth_years[subject_id]
                    result['age'] = target_examdate.year - birth_year
        
        # 6. 生成description
        result['description'] = generate_description(
            result['age'],
            result['sex'],
            result['weight']
        )
        
        results.append(result)
        
        if (idx + 1) % 100 == 0:
            print(f"  已处理 {idx + 1}/{len(df_pairs)} 行...")
    
    print(f"\n匹配成功: {matched_count}/{len(df_pairs)} 行")
    
    # 创建结果DataFrame
    df_result = pd.DataFrame(results)
    
    # 保存结果
    df_result.to_csv(output_path, index=False)
    print(f"\n结果已保存至: {output_path}")
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"总行数: {len(df_result)}")
    print(f"有sex信息: {df_result['sex'].notna().sum()}")
    print(f"有weight信息: {df_result['weight'].notna().sum()}")
    print(f"有diagnosis信息: {df_result['diagnosis'].notna().sum()}")
    print(f"有age信息: {df_result['age'].notna().sum()}")
    print(f"有完整description: {df_result['description'].notna().sum()}")
    
    # 显示前几行结果
    print("\n=== 前5行结果 ===")
    print(df_result.head().to_string())
    
    return df_result

if __name__ == '__main__':
    df = main()
