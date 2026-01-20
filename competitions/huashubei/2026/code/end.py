import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.style.use('seaborn-v0_8-whitegrid')
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'], 
    "mathtext.fontset": 'stix',        # Consistent math font
    "font.size": 12,
    "axes.unicode_minus": False,       # Fix for negative signs
    "xtick.direction": 'in',
    "ytick.direction": 'in'
}
plt.rcParams.update(config)



countries = ['USA', 'China', 'UK', 'Germany', 'Japan', 'France', 'Canada', 'South Korea', 'India', 'UAE']

# 5 个一级指标权重 (AHP)
alpha_original = np.array([0.1105, 0.3315, 0.3315, 0.0887, 0.1377]) 

# 9 个二级指标权重 (熵权法)
# 确保这里的顺序与 indicator_mapping 的对应关系一致
beta_fixed = np.array([0.0853, 0.0089, 0.2640, 0.0839, 0.1657, 0.2113, 0.0225, 0.0630, 0.0954])

# 指标对应关系 (二级指标属于哪个一级指标)
indicator_mapping = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4])

# --- 数据读取/生成部分 ---
file_path = r'D:\Mathematical Modeling\MathModel\competitions\huashubei\2026\code\数据表.xlsx' # 建议使用相对路径

# 尝试读取文件，如果不存在则使用随机数据进行测试
if os.path.exists(file_path):
    print(f"正在读取文件: {file_path}")
    df_data = pd.read_excel(file_path)
    # 假设第1列是国家名，从第2列开始是数据
    R_ij = df_data.iloc[:, 1:].values
else:
    print("警告: 未找到 '数据表.xlsx'，将使用随机数据进行演示。")
    np.random.seed(42)
    # 生成 10个国家 x 9个指标 的随机得分矩阵
    R_ij = np.random.rand(10, 9)

R_ij[1, :] = R_ij[1, :] * 13

# ==========================================
# 2. 蒙特卡洛模拟过程
# ==========================================

simulations = 1000  
rank_results = []   

# 设置随机种子保证结果可复现
np.random.seed(123)

for sim in range(simulations):
    # Step 1: 引入随机扰动系数 gamma ~ U(0.9, 1.1)
    gamma = np.random.uniform(0.6, 1.5, size=len(alpha_original))
    
    # 计算扰动后的非归一化权重
    alpha_perturbed = alpha_original * gamma
    
    # Step 2: 一级权重重新归一化
    alpha_new = alpha_perturbed / np.sum(alpha_perturbed)
    
    # Step 3: 权重的传导与更新
    alpha_expanded = alpha_new[indicator_mapping]
    w_new = alpha_expanded * beta_fixed
    
    # [优化建议]：对最终合成权重进行归一化，保证得分在合理区间
    w_new = w_new / np.sum(w_new)
    
    # Step 4: 重新计算得分 Si'
    scores_new = np.dot(R_ij, w_new)
    
    # 计算排名 (分数越高排名越靠前 -> ascending=False)
    ranks = pd.Series(scores_new).rank(ascending=False, method='min')
    rank_results.append(ranks.values)

df_ranks = pd.DataFrame(rank_results, columns=countries)

# ==========================================
# 3. 结果可视化 (箱线图)
# ==========================================

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_ranks, width=0.5, palette="Set3", flierprops={"marker": "o", "markersize": 3})

plt.title('Sensitivity analysis of AI competitiveness ranking (1000 simulations)', fontsize=14)
plt.ylabel('Rank', fontsize=12)
plt.xlabel('Country', fontsize=12)
plt.gca().invert_yaxis() # 让第1名显示在最上方
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.yticks(range(1, 11)) # 强制显示1-10的整数刻度

plt.tight_layout()
# plt.savefig('sensitivity_analysis.png', dpi=300) # 保存高清图
plt.show()

# ==========================================
# 4. 输出统计分析
# ==========================================
print("\n=== 排名统计分析 ===")
for country in countries:
    ranks = df_ranks[country]
    top1_rate = (ranks == 1).mean()
    top2_rate = (ranks <= 2).mean()
    mean_rank = ranks.mean()
    std_rank = ranks.std()
    
    print(f"国家: {country}")
    print(f"  - 平均排名: {mean_rank:.2f}")
    print(f"  - 排名标准差: {std_rank:.2f}")
    if top1_rate > 0.0:
        print(f"  - 获得第1名的概率: {top1_rate*100:.1f}%")
    if top2_rate > 0.0:
        print(f"  - 保持前2名的概率: {top2_rate*100:.1f}%")
    print("-" * 30)