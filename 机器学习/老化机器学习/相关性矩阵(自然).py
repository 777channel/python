import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# --- 1. 设置全局字体 ---
try:
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.weight'] = 'bold'
except:
    print("Warning: 'Times New Roman' font not found. Using default font.")


# --- 2. 读取数据 ---
file_path = r'C:\Users\AMDYE\OneDrive\python\imputation_output_all_features_stratified_single_plot_en\imputed_data_KNN_Imputation_all_features.csv'
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='gbk')

# --- 3. 清洗 Area 列，移除后缀 ---
# 使用正则表达式一次性移除所有可能的后缀，以确保能正确匹配
print("正在清洗 'Area' 列，移除 '省', '市', '自治区' 等后缀...")
df['Area'] = df['Area'].str.replace('省|市|自治区|维吾尔|回族|壮族', '', regex=True)
print("清洗完成。")


# --- 4. 进行 Area 列的转换 ---
province_to_region = {
    '辽宁': '东北', '吉林': '东北', '黑龙江': '东北',
    '北京': '华北', '天津': '华北', '河北': '华北', '山西': '华北', '内蒙古': '华北',
    '上海': '华东', '江苏': '华东', '浙江': '华东', '安徽': '华东', '福建': '华东', '江西': '华东', '山东': '华东',
    '河南': '华中', '湖北': '华中', '湖南': '华中',
    '广东': '华南', '广西': '华南', '海南': '华南',
    '重庆': '西南', '四川': '西南', '贵州': '西南', '云南': '西南', '西藏': '西南',
    '陕西': '西北', '甘肃': '西北', '青海': '西北', '宁夏': '西北', '新疆': '西北'
}
region_to_id = {
    '东北': 1, '华北': 2, '华东': 3, '华中': 4, '华南': 5, '西南': 6, '西北': 7
}

df['Region'] = df['Area'].map(province_to_region)
df['Area_ID'] = df['Region'].map(region_to_id)

# 健壮性检查
unmapped_count = df['Area_ID'].isnull().sum()
if unmapped_count > 0:
    unmapped_provinces = df[df['Area_ID'].isnull()]['Area'].unique()
    print("--- 警告 ---")
    print(f"清洗后，仍有 {unmapped_count} 行数据未能成功转换为区域ID。")
    print(f"未能识别的名称为: {list(unmapped_provinces)}")
    print("------------")

# --- 5. 排除列并准备数据 ---
df_for_corr = df.drop(columns=['Area', 'Region', 'T-Aging'])
numeric_df = df_for_corr.select_dtypes(include=np.number)

if numeric_df.shape[1] < 2:
    print("--- 错误 ---")
    print(f"数值列不足2个，无法生成相关性矩阵。找到的数值列: {list(numeric_df.columns)}")
    exit()

# ===================================================================
# --- 6. 【核心修改】将 'Area_ID' 移动到第一列，以确保它在图中显示在最前面 ---
feature_to_front = 'Area_ID'
if feature_to_front in numeric_df.columns:
    cols = numeric_df.columns.tolist()
    # 将 'Area_ID' 弹出并插入到列表的第一个位置
    cols.insert(0, cols.pop(cols.index(feature_to_front)))
    # 根据新的列顺序重新排列DataFrame
    numeric_df = numeric_df[cols]
    print(f"'{feature_to_front}' 特征已成功移动到第一列，用于绘图排序。")
else:
    print(f"警告：在数值列中未找到 '{feature_to_front}'。")
# ===================================================================


# --- 7. 计算相关性矩阵 ---
corr = numeric_df.corr(method='pearson')

# --- 8. 将 'Area_ID' 重命名为 'Area' 以便绘图 ---
corr = corr.rename(index={'Area_ID': 'Area'}, columns={'Area_ID': 'Area'})


# --- 9. 绘制相关性矩阵图 ---
fig, ax = plt.subplots(figsize=(18, 16), dpi=1200)
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=-1, vmax=1)

text_in_figure_size, axis_label_size, title_size = 22, 22, 24

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        if i > j:
            color = cmap(norm(corr.iloc[i, j]))
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                    ha='center', va='center', fontsize=text_in_figure_size,
                    fontweight='bold', color=color)
        elif i < j:
            color = cmap(norm(corr.iloc[i, j]))
            ax.scatter(j, i, s=np.abs(corr.iloc[i, j]) * 500, color=color, alpha=0.7)

ax.set_xticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90, ha='center', fontsize=axis_label_size)
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_yticklabels(corr.columns, fontsize=axis_label_size)

ax.set_xlim(-0.5, len(corr.columns) - 0.5)
ax.set_ylim(-0.5, len(corr.columns) - 0.5)
ax.invert_yaxis()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
# 【修改】将颜色条标签与计算方法统一
cbar.set_label('Pearson Correlation Coefficient', fontsize=24, fontweight='bold')
cbar.ax.tick_params(labelsize=22)

# ax.set_title('Pearson Correlation Matrix', fontsize=24, pad=20, fontweight='bold')
plt.tight_layout()


# --- 10. 保存图形 ---
# 【修改】将文件名与计算方法统一
output_filename = "Pearson_Correlation_Final_with_AreaName.pdf"
plt.savefig(output_filename, format='pdf', bbox_inches='tight')
plt.close(fig)

print(f"图形已成功保存为: {output_filename}")