import pandas as pd
from collections import defaultdict

def analyze_soil_data(file_path):
    """
    按照施炭比(1%-5%)和灌溉方式(干湿交替/持续淹水)分类统计土壤属性
    
    参数:
    file_path: CSV文件路径
    """
    # 读取CSV文件
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # 检查数据
    print(f"总数据量: {len(df)}")
    print(f"列名: {df.columns.tolist()}")
    
    # 数据预处理
    # 将pH和镉浓度四舍五入到一位小数
    df['土壤pH'] = df['土壤pH'].round(1)
    df['初始镉浓度(mg/kg)'] = df['初始镉浓度(mg/kg)'].round(1)
    # 将有机质含量和CEC四舍五入到整数
    df['有机质含量(g/kg)'] = df['有机质含量(g/kg)'].round().astype(int)
    df['CEC值'] = df['CEC值'].round().astype(int)

    # 统计结果存储
    results = {}
    
    # 按施炭比分类(1%-5%)
    for carbon_ratio in ['1%', '2%', '3%', '4%', '5%']:
        # 筛选该施炭比的数据
        ratio_df = df[df['施炭比'] == carbon_ratio]
        
        if len(ratio_df) == 0:
            print(f"警告: 未找到施炭比为{carbon_ratio}的数据")
            continue
            
        print(f"\n施炭比 {carbon_ratio} 的数据数量: {len(ratio_df)}")
        
        # 创建该施炭比的结果字典
        results[carbon_ratio] = {}
        
        # 按灌溉方式进一步分类
        for irrigation in ['干湿交替', '持续淹水']:
            # 筛选该灌溉方式的数据
            irr_df = ratio_df[ratio_df['灌溉方式'] == irrigation]
            
            if len(irr_df) == 0:
                print(f"  警告: 未找到{carbon_ratio}施炭比下{irrigation}的数据")
                continue
                
            print(f"  灌溉方式 {irrigation} 的数据数量: {len(irr_df)}")
            
            # 创建该灌溉方式的结果字典
            results[carbon_ratio][irrigation] = {
                'pH': {},
                '有机质': {},
                'CEC': {},
                '镉浓度': {}
            }
            
            # 统计pH值分布
            ph_counts = irr_df['土壤pH'].value_counts().sort_index()
            for ph, count in ph_counts.items():
                results[carbon_ratio][irrigation]['pH'][ph] = count
                
            # 输出每个具体pH值的统计
            print(f"\n    {irrigation}条件下的pH值分布:")
            for ph, count in sorted(ph_counts.items()):
                print(f"      pH {ph}: {count}个数据")
            
            # 统计有机质含量分布
            om_counts = irr_df['有机质含量(g/kg)'].value_counts().sort_index()
            for om, count in om_counts.items():
                results[carbon_ratio][irrigation]['有机质'][om] = count
                
            # 输出每个具体有机质含量的统计
            print(f"\n    {irrigation}条件下的有机质含量(g/kg)分布:")
            for om, count in sorted(om_counts.items()):
                print(f"      {om} g/kg: {count}个数据")
            
            # 统计CEC值分布
            cec_counts = irr_df['CEC值'].value_counts().sort_index()
            for cec, count in cec_counts.items():
                results[carbon_ratio][irrigation]['CEC'][cec] = count
                
            # 输出每个具体CEC值的统计
            print(f"\n    {irrigation}条件下的CEC值分布:")
            for cec, count in sorted(cec_counts.items()):
                print(f"      CEC {cec}: {count}个数据")
            
            # 统计初始镉浓度分布
            cd_counts = irr_df['初始镉浓度(mg/kg)'].value_counts().sort_index()
            for cd, count in cd_counts.items():
                results[carbon_ratio][irrigation]['镉浓度'][cd] = count
                
            # 输出每个具体镉浓度的统计
            print(f"\n    {irrigation}条件下的初始镉浓度(mg/kg)分布:")
            for cd, count in sorted(cd_counts.items()):
                print(f"      {cd} mg/kg: {count}个数据")
    
    return results

def print_results(results):
    """打印统计结果 - 并排表格形式"""
    for ratio, ratio_data in results.items():
        print(f"\n=================== 施炭比 {ratio} ===================")
        
        # 合并不同灌溉方式的pH值，便于并排比较
        print("\n  pH值比较 (干湿交替 vs 持续淹水):")
        # 获取所有可能的pH值
        all_ph_values = set()
        for irrigation, irr_data in ratio_data.items():
            all_ph_values.update(irr_data['pH'].keys())
        
        # 打印表头
        print("    pH值    |  干湿交替  |  持续淹水  ")
        print("    --------|-----------|----------")
        
        # 打印每个pH值的数据量
        for ph in sorted(all_ph_values):
            dry_wet_count = ratio_data.get('干湿交替', {}).get('pH', {}).get(ph, 0)
            flooded_count = ratio_data.get('持续淹水', {}).get('pH', {}).get(ph, 0)
            print(f"    pH {ph:4}  |    {dry_wet_count:3}    |    {flooded_count:3}    ")
        
        # 同样处理其他属性
        print("\n  有机质含量比较 (干湿交替 vs 持续淹水):")
        all_om_values = set()
        for irrigation, irr_data in ratio_data.items():
            all_om_values.update(irr_data['有机质'].keys())
        
        print("    有机质(g/kg) |  干湿交替  |  持续淹水  ")
        print("    ------------|-----------|----------")
        for om in sorted(all_om_values):
            dry_wet_count = ratio_data.get('干湿交替', {}).get('有机质', {}).get(om, 0)
            flooded_count = ratio_data.get('持续淹水', {}).get('有机质', {}).get(om, 0)
            print(f"    {om:12} |    {dry_wet_count:3}    |    {flooded_count:3}    ")
        
        print("\n  CEC值比较 (干湿交替 vs 持续淹水):")
        all_cec_values = set()
        for irrigation, irr_data in ratio_data.items():
            all_cec_values.update(irr_data['CEC'].keys())
        
        print("    CEC值      |  干湿交替  |  持续淹水  ")
        print("    ------------|-----------|----------")
        for cec in sorted(all_cec_values):
            dry_wet_count = ratio_data.get('干湿交替', {}).get('CEC', {}).get(cec, 0)
            flooded_count = ratio_data.get('持续淹水', {}).get('CEC', {}).get(cec, 0)
            print(f"    {cec:12} |    {dry_wet_count:3}    |    {flooded_count:3}    ")
        
        print("\n  镉浓度比较 (干湿交替 vs 持续淹水):")
        all_cd_values = set()
        for irrigation, irr_data in ratio_data.items():
            all_cd_values.update(irr_data['镉浓度'].keys())
        
        print("    镉浓度(mg/kg) |  干湿交替  |  持续淹水  ")
        print("    -------------|-----------|----------")
        for cd in sorted(all_cd_values):
            dry_wet_count = ratio_data.get('干湿交替', {}).get('镉浓度', {}).get(cd, 0)
            flooded_count = ratio_data.get('持续淹水', {}).get('镉浓度', {}).get(cd, 0)
            print(f"    {cd:13} |    {dry_wet_count:3}    |    {flooded_count:3}    ")

def plot_results(results, output_folder=None):
    """
    绘制统计结果图表，对比同一施炭比下不同灌溉方式的属性分布
    
    参数:
    results: 统计结果字典
    output_folder: 图表保存文件夹，不提供则不保存
    """
    properties = ['pH', '有机质', 'CEC', '镉浓度']
    property_labels = ['土壤pH', '有机质含量(g/kg)', 'CEC值', '初始镉浓度(mg/kg)']
    
    for ratio, ratio_data in results.items():
        for prop_idx, prop in enumerate(properties):
            plt.figure(figsize=(12, 7))
            
            # 收集所有可能的属性值
            all_values = set()
            for irrigation, irr_data in ratio_data.items():
                if prop in irr_data:
                    all_values.update(irr_data[prop].keys())
            
            all_values = sorted(all_values)
            
            # 收集每个灌溉方式的数据
            bar_width = 0.35
            index = np.arange(len(all_values))
            
            # 获取干湿交替的数据
            dry_wet_counts = []
            for val in all_values:
                count = ratio_data.get('干湿交替', {}).get(prop, {}).get(val, 0)
                dry_wet_counts.append(count)
            
            # 获取持续淹水的数据
            flooded_counts = []
            for val in all_values:
                count = ratio_data.get('持续淹水', {}).get(prop, {}).get(val, 0)
                flooded_counts.append(count)
            
            # 绘制条形图
            plt.bar(index - bar_width/2, dry_wet_counts, bar_width, label='干湿交替', alpha=0.7, color='#3498db')
            plt.bar(index + bar_width/2, flooded_counts, bar_width, label='持续淹水', alpha=0.7, color='#e74c3c')
            
            # 设置图表属性
            plt.xlabel(property_labels[prop_idx], fontsize=12)
            plt.ylabel('数据数量', fontsize=12)
            plt.title(f'施炭比 {ratio} - {property_labels[prop_idx]}分布对比', fontsize=14)
            plt.xticks(index, all_values)
            plt.legend(fontsize=11)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 在柱状图上添加数值标签
            for i, v in enumerate(dry_wet_counts):
                if v > 0:
                    plt.text(i - bar_width/2, v + 0.5, str(v), ha='center', fontsize=9)
            
            for i, v in enumerate(flooded_counts):
                if v > 0:
                    plt.text(i + bar_width/2, v + 0.5, str(v), ha='center', fontsize=9)
            
            # 保存图表
            if output_folder:
                filename = f"{output_folder}/施炭比{ratio}_{property_labels[prop_idx]}_对比.png"
                plt.savefig(filename, dpi=300)
                print(f"图表已保存: {filename}")
            else:
                plt.show()
            
            plt.close()

def main():
    """主函数"""
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\生物炭结果\1^1-5^1 data.csv"  # 数据文件路径
    
    try:
        import pandas as pd
    except ImportError:
        print("错误: 未安装pandas库。请先使用以下命令安装:")
        print("pip install pandas")
        return
    
    print("开始分析数据...")
    
    # 进行数据分析
    results = analyze_soil_data(file_path)
    
    # 打印统计结果
    print_results(results)
    
    print("数据分析完成。")

if __name__ == "__main__":
    main()