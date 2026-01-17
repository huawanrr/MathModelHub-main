import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("--- Step 1: AHP 层次分析法权重计算 ---")
    # 依据文档提供的判断矩阵
    # 顺序: A1基础设施, A2人才与教育, A3科研与创新, A4数据与应用, A5国家政策与资本
    A = np.array([
        [1,   1/3, 1/3, 1,   1],
        [3,   1,   1,   3,   3],
        [3,   1,   1,   3,   3],
        [1,   1/3, 1/3, 1,   1],
        [1,   1/3, 1/3, 3,   1]
    ])

    n = 5
    # 1. 几何平均法求权重 
    product_rows = np.prod(A, axis=1)
    g_i = np.power(product_rows, 1/n)
    w_ahp = g_i / np.sum(g_i)
    
    print("AHP 一级指标权重 (w_ahp):")
    indicator_names = ['A1 基础设施', 'A2 人才教育', 'A3 科研创新', 'A4 数据应用', 'A5 政策资本']
    for name, w in zip(indicator_names, w_ahp):
        print(f"{name}: {w:.4f}")

    # 2. 一致性检验 
    Aw = np.dot(A, w_ahp)
    lambda_max = np.mean(Aw / w_ahp)
    CI = (lambda_max - n) / (n - 1)
    RI = 1.12  # n=5 时的 RI 值
    CR = CI / RI

    print(f"\n最大特征值 lambda_max: {lambda_max:.4f}")
    print(f"一致性指标 CI: {CI:.4f}")
    print(f"一致性比例 CR: {CR:.4f}")

    if CR < 0.10:
        print(" 一致性检验通过 (CR < 0.10)，无需调整矩阵。\n")
    else:
        print(" 一致性检验未通过 (CR >= 0.10)，请微调判断矩阵！\n")

    print("-" * 30)
    print("--- Step 2 & 3: 数据加载、标准化与熵权法 ---")

    # 注意要在引号前加 r，防止路径中的斜杠被转义
    df = pd.read_excel(r'd:\Mathematical Modeling\MathModel\competitions\huashubei\2026\code\数据表.xlsx', index_col=0,header=1)
    cols = df.columns.tolist()
    
    print("原始数据预览 (前5行):")
    print(df.head())
    print("-" * 20)
    # ============================================

    # 1. 极值标准化 
    # x_norm = (x - min) / (max - min)
    # 计算极差，如果为0则替换为1，防止除以零
    range_val = df.max() - df.min()
    range_val[range_val == 0] = 1 

    df_norm = (df - df.min()) / range_val

    
    # 2. 熵权法计算 
    # 计算比重 p_ij
    p_ij = df_norm / df_norm.sum()
    # 避免 log(0) 的情况，加上一个极小值
    k = 1 / np.log(len(df))
    e_j = -k * (p_ij * np.log(p_ij + 1e-10)).sum()
    d_j = 1 - e_j
    w_entropy = d_j / d_j.sum()
    
    print("熵权法二级指标权重 (w_entropy):")
    for col, w in zip(cols, w_entropy):
        print(f"{col}: {w:.4f}")

    print("-" * 30)
    print("--- Step 4: 组合权重与综合评分 ---")

    # 组合权重计算 
    # w_j = w_AHP(对应的一级) * w_Entropy(自身)
    # 映射关系：前2个二级指标属于A1，接下来的2个属于A2...
    w_combined = []
    
    # 将 AHP 权重扩展到对应的二级指标 
    ahp_expanded = []
    for w in w_ahp:
        ahp_expanded.extend([w, w]) # 每个一级指标重复2次
    
    w_combined = np.array(ahp_expanded) * w_entropy
    
    # 注意：文档公式是直接相乘。通常为了保证最终得分为0-1区间，可以对组合权重归一化
    # 但严格遵循公式 则直接使用。
    # 这里我们做一个归一化处理，方便解释，如果不想要归一化可以注释掉下面这行
    w_combined = w_combined / w_combined.sum()

    print("最终组合权重 (Normalized):")
    weight_dict = dict(zip(cols, w_combined))
    for col, w in weight_dict.items():
        print(f"{col}: {w:.4f}")

    # 计算综合得分 Si 
    # 得分 = 标准化矩阵 * 组合权重向量
    scores = df_norm.dot(w_combined)
    df_result = pd.DataFrame({'Score': scores}).sort_values('Score', ascending=False)
    
    print("\n各国 AI 发展能力综合排名:")
    print(df_result)
    
    print("-" * 30)
    print("--- Step 5: Spearman 相关性分析 ---")
    
    # 计算 Spearman 相关系数矩阵
    corr_matrix = df_norm.corr(method='spearman')
    print("Spearman 相关系数矩阵数值表:")
    # 使用 pandas 的 option_context 临时设置显示格式，保证对齐和精度
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000, 'display.float_format', '{:,.4f}'.format):
        print(corr_matrix)
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Spearman correlation heatmap for secondary indicators')
    plt.tight_layout()
    plt.show()
    
    print("相关性分析完成，热力图已生成。")

if __name__ == "__main__":
    main()