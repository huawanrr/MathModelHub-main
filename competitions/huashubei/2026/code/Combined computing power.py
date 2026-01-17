import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================

# 1. 设置数据文件夹路径（建议使用绝对路径，防止找不到文件）
# 请将下面的路径改为你实际存放文件的文件夹路径
data_folder_path = r'D:\Mathematical Modeling\MathModel\competitions\huashubei\2026\data'  

# 2. 文件列表：请准确填写每个年份对应的文件名（注意后缀是 .xls 还是 .xlsx）
file_map = {
    'TOP500_201611.xls': 2016,  # 注意这里是 .xls
    'TOP500_201711.xls': 2017,
    'TOP500_201811.xls': 2018,
    'TOP500_201911.xls': 2019,
    'TOP500_202011.xlsx': 2020, # 从这里开始可能是 .xlsx，请根据实际情况修改
    'TOP500_202111.xlsx': 2021,
    'TOP500_202211.xlsx': 2022,
    'TOP500_202311.xlsx': 2023,
    'TOP500_202411.xlsx': 2024,
    'TOP500_202511.xlsx': 2025
}

# 3. 目标国家
target_countries = [
    'United States', 'China', 'United Kingdom', 'Germany', 
    'Korea, South', 'Japan', 'France', 'Canada', 
    'United Arab Emirates', 'India'
]

# ================= 处理逻辑 =================

yearly_sums = {}

print(f"准备从 {data_folder_path} 读取数据...\n")

for filename, year in file_map.items():
    full_path = os.path.join(data_folder_path, filename)
    
    try:
        # --- 核心修改：根据后缀自动选择读取引擎 ---
        if filename.lower().endswith('.xls'):
            # 读取 .xls (2016-2019)
            # 如果报错 "No module named 'xlrd'"，请在终端运行: pip install xlrd
            df = pd.read_excel(full_path, engine='xlrd')
        else:
            # 读取 .xlsx (2020-2025)
            df = pd.read_excel(full_path, engine='openpyxl')

        # --- 以下是标准处理流程 ---
        
        # 1. 寻找 Rmax 列
        possible_cols = [c for c in df.columns if 'Rmax' in str(c)]
        
        if possible_cols:
            rmax_col = possible_cols[0]
            
            # 2. 清洗数据（转数字，处理非法字符）
            df[rmax_col] = pd.to_numeric(df[rmax_col], errors='coerce').fillna(0)
            
            
            
            # 4. 按国家求和
            if 'Country' in df.columns:
                df['Country'] = df['Country'].astype(str).str.strip() # 去除空格
                
                # 计算总和并转换单位
                sums = df.groupby('Country')[rmax_col].sum() 
                yearly_sums[year] = sums
                
                print(f"✅ {year}年 ({filename}) 读取成功")
            else:
                print(f"⚠️ {year}年 ({filename}) 缺少 'Country' 列，跳过")
        else:
            print(f"⚠️ {year}年 ({filename}) 缺少 'Rmax' 列，跳过")
            
    except FileNotFoundError:
        print(f"❌ 文件不存在: {full_path}")
    except ImportError as e:
        print(f"❌ 缺少必要的库: {e} (请尝试运行 pip install xlrd openpyxl)")
    except Exception as e:
        print(f"❌ 处理 {filename} 时出错: {e}")

# ================= 保存结果 =================

if yearly_sums:
    # 合并数据
    df_all = pd.DataFrame(yearly_sums).fillna(0)
    
    # 计算 10 年平均值
    df_all['Average_TFlops'] = df_all.mean(axis=1)
    
    # 筛选目标国家并排序
    final_result = df_all.reindex(target_countries).fillna(0)
    final_result = final_result.sort_values(by='Average_TFlops', ascending=False)
    
    # 保存
    output_file = 'Country_Computing_Power_Final.xlsx'
    final_result.to_excel(output_file)
    print(f"\n🎉 处理完成！结果已保存为: {output_file}")
    print("\n预览结果 (Average_TFlops):")
    print(final_result[['Average_TFlops']])
else:
    print("\n没有生成任何数据，请检查路径和文件名。")