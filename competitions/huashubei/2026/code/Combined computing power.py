import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================

# 1. 设置数据文件夹路径
# 请确保路径正确，并且文件夹里有 TOP500_202511.xlsx 这个文件
data_folder_path = r'D:\Mathematical Modeling\MathModel\competitions\huashubei\2026\data' 

# 2. 文件列表：只保留 2025 年
file_map = {
    'TOP500_202511.xlsx': 2025
}

# 3. 目标国家
target_countries = [
    'United States', 'China', 'United Kingdom', 'Germany', 
    'South Korea', 'Japan', 'France', 'Canada', 
    'United Arab Emirates', 'India'
]

# ================= 处理逻辑 =================

yearly_sums = {}

print(f"准备从 {data_folder_path} 读取 2025 年数据...\n")

for filename, year in file_map.items():
    full_path = os.path.join(data_folder_path, filename)
    
    try:
        # 读取 .xlsx (2025)
        # engine='openpyxl' 专门用于读取 xlsx 文件
        df = pd.read_excel(full_path, engine='openpyxl')

        # 1. 寻找 Rmax 列
        possible_cols = [c for c in df.columns if 'Rmax' in str(c)]
        
        if possible_cols:
            rmax_col = possible_cols[0]
            
            # 2. 清洗数据（转数字，处理非法字符）
            df[rmax_col] = pd.to_numeric(df[rmax_col], errors='coerce').fillna(0)
            
            # 3. 按国家求和
            if 'Country' in df.columns:
                df['Country'] = df['Country'].astype(str).str.strip() # 去除空格
                
                # 计算总和
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
        print(f"❌ 缺少必要的库: {e} (请确保安装了 openpyxl)")
    except Exception as e:
        print(f"❌ 处理 {filename} 时出错: {e}")

# ================= 保存结果 =================

if yearly_sums:
    # 合并数据 (此时只有一列 2025)
    df_all = pd.DataFrame(yearly_sums).fillna(0)
    
    # 筛选目标国家
    final_result = df_all.reindex(target_countries).fillna(0)
    
    # 按照 2025 年的数据列降序排序
    # 注意：这里的列名是整数 2025
    if 2025 in final_result.columns:
        final_result = final_result.sort_values(by=2025, ascending=False)
        # 重命名列名，让结果更清晰
        final_result.rename(columns={2025: 'Total_Rmax_2025'}, inplace=True)
    
    # 保存
    output_file = 'Country_Computing_Power_2025_Only.xlsx'
    final_result.to_excel(output_file)
    
    print(f"\n🎉 处理完成！结果已保存为: {output_file}")
    print("\n预览结果 (Total_Rmax_2025):")
    print(final_result)
else:
    print("\n没有生成任何数据，请检查路径和文件名。")