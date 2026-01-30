"""
生成包含人口统计学信息的CSV表格
字段：subject_id, examdate, sex, weight, diagnosis, age, description, MMSE, CDR, GDS, FAQ, NPI-Q, old_descr
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


def aggregate_clinical_values(df_mytable_subject, target_examdate, target_diagnosis):
    """
    从诊断一致的所有行中聚合临床评估字段（MMSE, CDR, GDS, FAQ, NPI-Q）。
    对于每个字段，选取examdate最接近的有效值。
    
    Args:
        df_mytable_subject: 同一subject的my_table行
        target_examdate: 目标examdate
        target_diagnosis: 目标诊断值（原始数值，如1/2/3）
    
    Returns:
        dict: 包含各临床字段值的字典
    """
    clinical_fields = {
        'MMSE': 'MMSCORE',
        'CDR': 'CDGLOBAL',
        'GDS': 'GDTOTAL',
        'FAQ': 'FAQTOTAL',
        'NPI-Q': 'NPISCORE'
    }
    
    result = {field: None for field in clinical_fields.keys()}
    
    if df_mytable_subject.empty or target_examdate is None:
        return result
    
    # 筛选诊断一致的行
    same_diag_rows = []
    for idx, row in df_mytable_subject.iterrows():
        row_diag = row.get('DIAGNOSIS')
        # 处理诊断值比较（可能是浮点数或整数）
        try:
            if pd.notna(row_diag) and pd.notna(target_diagnosis):
                if int(float(row_diag)) == int(float(target_diagnosis)):
                    same_diag_rows.append(row)
        except:
            continue
    
    if not same_diag_rows:
        return result
    
    # 对每个临床字段，找到examdate最接近且有值的行
    for field_name, col_name in clinical_fields.items():
        min_diff = float('inf')
        best_value = None
        
        for row in same_diag_rows:
            row_examdate = row.get('calculated_examdate')
            if row_examdate is None:
                continue
            
            # 获取字段值
            value = row.get(col_name)
            if pd.isna(value) or value == '':
                continue
            
            try:
                value = float(value)
            except:
                continue
            
            # 计算日期差
            diff = abs((row_examdate - target_examdate).days)
            if diff < min_diff:
                min_diff = diff
                best_value = value
        
        result[field_name] = best_value
    
    return result


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


def get_clinical_value(value):
    """获取临床评估值，处理缺失值"""
    if pd.isna(value) or value == '':
        return None
    try:
        return float(value)
    except:
        return None


def generate_old_descr(age, sex, weight, cdr, mmse, gds, faq, npiq):
    """
    生成old_descr文本，格式：
    "Subject is a <age>-year-old <sex> with a weight of <weight> kg.
    The global Clinical Dementia Rating (CDR) score, which assesses dementia severity (0: no dementia to 3: severe dementia), is <CDR>.
    The Mini-Mental State Examination (MMSE) score, assessing cognitive function (0: severe impairment to 30: normal), is <MMSE>.
    The Geriatric Depression Scale (GDS) score, screening depression (0: no depression to 15: severe depression), is <GDS>.
    The Functional Activities Questionnaire (FAQ) score, assessing daily activity impairment (0: no impairment to 30: severe impairment), is <FAQ>.
    The Neuropsychiatric Inventory Questionnaire (NPI-Q) Total Score, assessing neuropsychiatric symptom burden (0: no symptoms to higher scores indicating greater burden), is <NPI-Q>"
    """
    # 基本信息部分
    if age is not None and sex is not None and weight is not None:
        base = f"Subject is a {age}-year-old {sex} with a weight of {weight} kg."
    elif age is not None and sex is not None:
        base = f"Subject is a {age}-year-old {sex}."
    elif age is not None:
        base = f"Subject is {age} years old."
    else:
        return None

    # 格式化临床评估值（整数或一位小数）
    def format_value(val):
        if val is None:
            return None
        if val == int(val):
            return str(int(val))
        else:
            return str(val)

    cdr_str = format_value(cdr)
    mmse_str = format_value(mmse)
    gds_str = format_value(gds)
    faq_str = format_value(faq)
    npiq_str = format_value(npiq)

    # 构建完整描述
    parts = [base]

    if cdr_str is not None:
        parts.append(f"The global Clinical Dementia Rating (CDR) score, which assesses dementia severity (0: no dementia to 3: severe dementia), is {cdr_str}.")

    if mmse_str is not None:
        parts.append(f"The Mini-Mental State Examination (MMSE) score, assessing cognitive function (0: severe impairment to 30: normal), is {mmse_str}.")

    if gds_str is not None:
        parts.append(f"The Geriatric Depression Scale (GDS) score, screening depression (0: no depression to 15: severe depression), is {gds_str}.")

    if faq_str is not None:
        parts.append(f"The Functional Activities Questionnaire (FAQ) score, assessing daily activity impairment (0: no impairment to 30: severe impairment), is {faq_str}.")

    if npiq_str is not None:
        parts.append(f"The Neuropsychiatric Inventory Questionnaire (NPI-Q) Total Score, assessing neuropsychiatric symptom burden (0: no symptoms to higher scores indicating greater burden), is {npiq_str}.")

    return " ".join(parts)

def main():
    # 设置路径
    base_path = '/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/data_csv'
    pairs_path = os.path.join(base_path, 'pairs_withPlasma.csv')
    mytable_path = os.path.join(base_path, 'All_Subjects_My_Table_25Jan2026.csv')
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
            'description': None,
            'MMSE': None,
            'CDR': None,
            'GDS': None,
            'FAQ': None,
            'NPI-Q': None,
            'old_descr': None
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
                target_diagnosis_raw = closest_row.get('DIAGNOSIS')
                result['diagnosis'] = convert_diagnosis(target_diagnosis_raw)

                # 计算年龄
                if subject_id in subject_birth_years and target_examdate is not None:
                    birth_year = subject_birth_years[subject_id]
                    result['age'] = target_examdate.year - birth_year

                # 6. 获取临床评估值（从诊断一致的多行中聚合）
                clinical_values = aggregate_clinical_values(
                    df_subject, target_examdate, target_diagnosis_raw
                )
                result['MMSE'] = clinical_values['MMSE']
                result['CDR'] = clinical_values['CDR']
                result['GDS'] = clinical_values['GDS']
                result['FAQ'] = clinical_values['FAQ']
                result['NPI-Q'] = clinical_values['NPI-Q']
        
        # 6. 生成description
        result['description'] = generate_description(
            result['age'],
            result['sex'],
            result['weight']
        )

        # 7. 生成old_descr
        result['old_descr'] = generate_old_descr(
            result['age'],
            result['sex'],
            result['weight'],
            result['CDR'],
            result['MMSE'],
            result['GDS'],
            result['FAQ'],
            result['NPI-Q']
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
    print(f"有MMSE信息: {df_result['MMSE'].notna().sum()}")
    print(f"有CDR信息: {df_result['CDR'].notna().sum()}")
    print(f"有GDS信息: {df_result['GDS'].notna().sum()}")
    print(f"有FAQ信息: {df_result['FAQ'].notna().sum()}")
    print(f"有NPI-Q信息: {df_result['NPI-Q'].notna().sum()}")
    print(f"有old_descr: {df_result['old_descr'].notna().sum()}")
    
    # 显示前几行结果
    print("\n=== 前5行结果 ===")
    print(df_result.head().to_string())
    
    return df_result

if __name__ == '__main__':
    df = main()
