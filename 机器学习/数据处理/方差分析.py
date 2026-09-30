import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import StandardScaler

def analyze_variance(file_path):
    try:
        # 读取数据
        # 如果是Excel文件
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        # 如果是CSV文件
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("不支持的文件格式，请使用.xlsx、.xls或.csv格式")

        # 检查数据是否包含缺失值
        if df.isnull().any().any():
            print("警告：数据中包含缺失值，将自动删除包含缺失值的行")
            df = df.dropna()

        # 获取因变量和自变量的列名
        print("\n可用的列名：", df.columns.tolist())
        y_col = input("请输入因变量的列名：")
        x_cols = input("请输入自变量的列名（用逗号分隔）：").split(',')
        x_cols = [x.strip() for x in x_cols]  # 移除可能的空格

        # 构建公式
        formula = f"{y_col} ~ " + " + ".join(x_cols)
        print(f"\n使用的模型公式：{formula}")

        # 进行方差分析
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)
        print("\n方差分析结果:")
        print(anova_table)

        # 标准化变量以比较相对重要性
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[x_cols])
        Y_scaled = scaler.fit_transform(df[[y_col]])

        # 使用标准化后的数据拟合模型
        df_scaled = pd.DataFrame(X_scaled, columns=x_cols)
        df_scaled[y_col] = Y_scaled
        model_scaled = ols(formula, data=df_scaled).fit()

        print("\n标准化系数（反映变量重要性）:")
        for var, coef in zip(x_cols, model_scaled.params[1:]):
            print(f"{var}: {coef:.4f}")

        # 计算并打印R方值
        print(f"\nR方值: {model.rsquared:.4f}")
        print(f"调整R方值: {model.rsquared_adj:.4f}")

    except Exception as e:
        print(f"分析过程中出现错误：{str(e)}")

# 使用示例
if __name__ == "__main__":
    file_path = input("请输入数据文件路径：")
    analyze_variance(file_path)