import numpy as np
import pandas as pd
import pickle
import os
import itertools
import warnings
warnings.filterwarnings('ignore')

# 定义统计参数
CEC_VALUES = [10, 15, 20]
CEC_MEAN = np.mean(CEC_VALUES)  # 15
CEC_STD = np.std(CEC_VALUES)    # 约4.08

def generate_input_combinations():
    """
    生成所有可能的输入组合
    """
    # 定义输入数据的可能取值范围
    input_ranges = {
        'pHsoil': [5.0, 5.5, 6.0, 6.5, 7.0],
        'organicmatter': [10, 15, 20, 25, 30],
        'Cdcontent': [0.6, 1.0, 2.0, 3.0, 4.0],
        'CEC': [10,15,20]
    }
    
    # 使用itertools.product生成笛卡尔积
    combinations = list(itertools.product(
        input_ranges['pHsoil'], 
        input_ranges['organicmatter'], 
        input_ranges['Cdcontent']
    ))
    
    # 转换为字典列表
    input_data = []
    for combo in combinations:
        for cec in CEC_VALUES:
            input_data.append({
                'pHsoil': combo[0], 
                'organicmatter': combo[1], 
                'Cdcontent': combo[2],
                'CEC': cec
            })
    
    return input_data

def normalize_input_data(data):
    """
    按照实际计算的统计参数进行归一化处理
    """
    normalized_data = []
    
    for item in data:
        norm_item = {}
        
        # pHsoil使用Min-Max归一化 [0,1]
        # 实际计算的范围：最小5.0，最大7.0
        norm_item['pHsoil'] = (item['pHsoil'] - 5.0) / (7.0 - 5.0)
        
        # organicmatter使用Z-score标准化
        # 实际计算的均值：20，标准差：7.9057
        norm_item['organicmatter'] = (item['organicmatter'] - 20) / 7.9057
        
        # Cdcontent使用对数变换+标准化
        # 确保值为正
        cdcontent = max(0.01, item['Cdcontent'])
        log_cd = np.log(cdcontent)
        # 对数转换后均值：0.5365，标准差：0.9216
        norm_item['log_Cdcontent'] = (log_cd -  0.5334) / 0.6996
        
        # CEC归一化 - 使用自身的均值和标准差
        norm_item['CEC'] = (item['CEC'] - CEC_MEAN) / CEC_STD
        
        # variety binary设为1（普通品种），不需要归一化
        norm_item['variety_binary'] = 1.0
        
        normalized_data.append(norm_item)
    
    return normalized_data

def load_first_model():
    """
    加载第一个预测模型
    """
    try:
        # 加载模型1：预测未施加生物炭时的稻米镉积累量
        with open(r'C:\Users\AMDYE\OneDrive\python\results\brown rice\models\CD_brown_rice_GBDT_model.pkl', 'rb') as f:
            model1_data = pickle.load(f)
            
            # 检查是否是字典类型，如果是则提取实际模型
            if isinstance(model1_data, dict) and 'model' in model1_data:
                model1 = model1_data['model']
            else:
                model1 = model1_data
        
        return model1
    
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return None

def predict_accumulation(model, normalized_data, irrigation_type):
    """
    预测未施加生物炭时的稻米镉积累量
    
    参数:
    model: 预测模型
    normalized_data: 归一化后的数据
    irrigation_type: 灌溉方式 ('alternating' 或 'continuous')
    
    返回:
    预测的镉积累量数组（归一化后的值）
    """
    # 将灌溉方式编码为数值
    irrigation_encoded = 1 if irrigation_type == "alternating" else 0
    
    # 准备模型输入 - 确保有6个特征
    X = np.zeros((len(normalized_data), 6))
    
    for i, item in enumerate(normalized_data):
        # 已有特征
        X[i, 0] = item['log_Cdcontent']  # log Cdcontent
        X[i, 1] = item['organicmatter']   # organicmatter
        X[i, 2] = irrigation_encoded      # irrigation type
        
        # 根据要求添加的特征
        X[i, 3] = item['CEC']  # 归一化后的CEC值
        X[i, 4] = item['pHsoil']          # pHsoil
        X[i, 5] = 1.0                     # variety binary（1表示普通品种）
    
    # 预测 - 返回归一化后的预测值
    return model.predict(X)

def reverse_normalize_prediction(normalized_predictions):
    """
    使用Z-score反归一化预测结果
    """
    # 反标准化
    real_predictions = normalized_predictions * 1.212 + 0.7671
    
    return real_predictions

def create_prediction_table(data, predictions, irrigation_type):
    """
    创建预测结果表格
    """
    # 创建结果列表
    results = []
    
    for i, (item, prediction) in enumerate(zip(data, predictions)):
        result = {
            '土壤pH': item['pHsoil'],
            '有机质含量(g/kg)': item['organicmatter'],
            '初始镉浓度(mg/kg)': item['Cdcontent'],
            'CEC值': item['CEC'],
            '预测镉积累量(mg/kg)': round(prediction, 3),
            '灌溉方式': "干湿交替" if irrigation_type == "alternating" else "持续淹水"
        }
        
        results.append(result)
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    return df

def main():
    """
    主函数
    """
    print("稻米镉积累量预测系统启动...")
    
    # 加载模型
    model1 = load_first_model()
    
    if model1 is None:
        print("模型加载失败，无法继续预测")
        return
    
    # 生成所有可能的输入组合
    input_data = generate_input_combinations()
    
    # 打印组合数量
    print(f"总输入组合数量: {len(input_data)}")
    
    # 归一化输入数据
    normalized_data = normalize_input_data(input_data)
    
    # 为两种灌溉方式分别创建预测
    irrigation_types = ["alternating", "continuous"]
    
    for irrigation_type in irrigation_types:
        print(f"\n处理 {irrigation_type} 灌溉方式...")
        
        # 预测镉积累量（归一化后的值）
        normalized_predictions = predict_accumulation(model1, normalized_data, irrigation_type)
        
        # 还原预测值
        predictions = reverse_normalize_prediction(normalized_predictions)
        
        # 创建预测表格
        predictions_df = create_prediction_table(input_data, predictions, irrigation_type)
        
        # 保存到CSV
        irrigation_name = "干湿交替灌溉" if irrigation_type == "alternating" else "持续淹水灌溉"
        os.makedirs('output', exist_ok=True)
        file_path = f"output/{irrigation_name}镉积累量预测.csv"
        predictions_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        print(f"结果已保存到: {file_path}")
        
        # 显示前几行表格
        print(f"\n{irrigation_name}预测结果(前10行):")
        print(predictions_df.head(10))
        
        # 打印预测值的统计特征
        print("\n预测值统计特征:")
        print("均值:", np.mean(predictions))
        print("标准差:", np.std(predictions))
        print("最小值:", np.min(predictions))
        print("最大值:", np.max(predictions))
    
    print("\n处理完成!")

if __name__ == "__main__":
    main()