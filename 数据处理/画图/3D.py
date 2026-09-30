import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

def extract_data_from_figure(image_path, left_crop=(50, 100, 380, 500), right_crop=(400, 100, 730, 500)):
    """
    从包含两个子图的单个图像中提取数据
    
    参数:
        image_path: 图像文件路径
        left_crop: 左图裁剪区域 (x_min, y_min, x_max, y_max)
        right_crop: 右图裁剪区域 (x_min, y_min, x_max, y_max)
    
    返回:
        两个DataFrame: 一个包含Cd吸附数据，一个包含As吸附数据
    """
    # 读取图像
    img = np.array(Image.open(image_path))
    
    # 裁剪左图 (Cd吸附)
    cd_img = img[left_crop[1]:left_crop[3], left_crop[0]:left_crop[2]]
    
    # 裁剪右图 (As吸附)
    as_img = img[right_crop[1]:right_crop[3], right_crop[0]:right_crop[2]]
    
    # 显示裁剪后的图像
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cd_img)
    axes[0].set_title('Cd(II)吸附图 (a)')
    axes[0].axis('off')
    
    axes[1].imshow(as_img)
    axes[1].set_title('As(III)吸附图 (b)')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    
    # 提取Cd吸附数据
    cd_data = process_image(
        cd_img, 
        x_label='As(III) Ce (mg/L)', 
        y_label='Cd(II) Ce (mg/L)', 
        z_label='Cd(II) qe (mg/L)',
        x_range=(0, 80),
        y_range=(0, 50),
        z_range=(0, 70),
        color_ranges=[
            ((0, 0, 100), (0, 0, 128), (0, 10)),     # 深蓝 <10
            ((0, 0, 200), (0, 0, 255), (10, 20)),    # 蓝色 10-20
            ((0, 80, 0), (0, 120, 80), (20, 30)),    # 深绿 20-30
            ((0, 150, 0), (100, 200, 0), (30, 40)),  # 绿色 30-40
            ((150, 200, 0), (200, 255, 0), (40, 50)), # 黄绿 40-50
            ((240, 200, 0), (255, 220, 0), (50, 60)), # 黄色 50-60
            ((255, 150, 0), (255, 0, 0), (60, 70))   # 红色 60-70
        ],
        num_clusters=7,
        title='Cd(II)吸附数据'
    )
    
    # 提取As吸附数据
    as_data = process_image(
        as_img, 
        x_label='Cd(II) Ce (mg/L)', 
        y_label='As(III) Ce (mg/L)', 
        z_label='As(III) qe (mg/L)',
        x_range=(0, 60),
        y_range=(0, 80),
        z_range=(0, 90),
        color_ranges=[
            ((0, 0, 100), (0, 0, 128), (0, 10)),     # 深蓝 <10
            ((0, 0, 200), (0, 0, 255), (10, 20)),    # 蓝色 10-20
            ((0, 80, 0), (0, 120, 80), (20, 30)),    # 深绿 20-30
            ((0, 150, 0), (100, 200, 0), (30, 40)),  # 绿色 30-40
            ((50, 180, 50), (150, 230, 50), (40, 50)), # 亮绿 40-50
            ((150, 200, 0), (200, 255, 0), (50, 60)), # 黄绿 50-60
            ((240, 200, 0), (255, 220, 0), (60, 70)), # 黄色 60-70
            ((255, 150, 0), (255, 180, 0), (70, 80)), # 橙色 70-80
            ((255, 100, 0), (255, 0, 0), (80, 90))   # 红色 80-90
        ],
        num_clusters=9,
        title='As(III)吸附数据'
    )
    
    # 合并数据
    merged_data = []
    
    # 创建网格点 (每0.5mg/L一个数据点)
    cd_concentrations = np.arange(0, 60.5, 0.5)  # Cd(II)初始浓度
    as_concentrations = np.arange(0, 80.5, 0.5)  # As(III)初始浓度
    
    # 对所有网格点结合cd_data和as_data
    for cd_conc in cd_concentrations:
        for as_conc in as_concentrations:
            # 找最接近的Cd数据点
            cd_subset = cd_data[(np.abs(cd_data['Cd(II) Ce (mg/L)'] - cd_conc) < 0.5) & 
                              (np.abs(cd_data['As(III) Ce (mg/L)'] - as_conc) < 0.5)]
            
            # 找最接近的As数据点
            as_subset = as_data[(np.abs(as_data['Cd(II) Ce (mg/L)'] - cd_conc) < 0.5) & 
                              (np.abs(as_data['As(III) Ce (mg/L)'] - as_conc) < 0.5)]
            
            cd_qe = np.nan
            if not cd_subset.empty:
                cd_qe = cd_subset['Cd(II) qe (mg/L)'].mean()
                
            as_qe = np.nan
            if not as_subset.empty:
                as_qe = as_subset['As(III) qe (mg/L)'].mean()
            
            # 如果至少有一个数据点有效，添加到合并数据
            if not np.isnan(cd_qe) or not np.isnan(as_qe):
                merged_data.append({
                    'Cd(II) Ce (mg/L)': cd_conc,
                    'As(III) Ce (mg/L)': as_conc,
                    'Cd(II) qe (mg/L)': cd_qe,
                    'As(III) qe (mg/L)': as_qe
                })
    
    # 创建合并后的DataFrame
    merged_df = pd.DataFrame(merged_data)
    
    return cd_data, as_data, merged_df

def process_image(img, x_label, y_label, z_label, x_range, y_range, z_range, 
                  color_ranges, num_clusters, title):
    """
    处理单个图像并提取数据
    
    参数:
        img: 图像数据
        x_label, y_label, z_label: 坐标轴标签
        x_range, y_range, z_range: 坐标轴范围
        color_ranges: 颜色范围到z值的映射列表
        num_clusters: 聚类数量
        title: 图像标题
    
    返回:
        包含提取数据的DataFrame
    """
    # 图像尺寸
    height, width = img.shape[:2]
    
    # 使用K-means进行颜色聚类
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.astype(int)
    
    # 显示识别出的颜色
    plt.figure(figsize=(12, 3))
    plt.suptitle(f'{title}中识别出的主要颜色')
    
    for i, center in enumerate(centers):
        plt.subplot(1, num_clusters, i+1)
        color_patch = np.ones((100, 100, 3), dtype=np.uint8)
        color_patch[:, :] = center
        plt.imshow(color_patch)
        plt.title(f'颜色 {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # 创建颜色到值的映射函数
    def get_z_value(color):
        min_dist = float('inf')
        z_val = 0
        
        for (color_min, color_max, z_range) in color_ranges:
            # 检查颜色是否在范围内
            in_range = all(c_min <= c <= c_max for c, c_min, c_max in zip(color, color_min, color_max))
            
            if in_range:
                # 颜色到范围中心的距离
                center_color = [(c_min + c_max) / 2 for c_min, c_max in zip(color_min, color_max)]
                dist = sum((c - cc) ** 2 for c, cc in zip(color, center_color))
                
                if dist < min_dist:
                    min_dist = dist
                    z_val = (z_range[0] + z_range[1]) / 2
                    
        return z_val

    # 生成均匀网格
    grid_size = 100
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    
    # 对每个网格点提取数据
    data = []
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算图像上的坐标
            img_x = int(j / grid_size * width)
            img_y = int((grid_size - i - 1) / grid_size * height)  # 翻转y坐标
            
            # 确保坐标在图像范围内
            img_x = min(max(0, img_x), width - 1)
            img_y = min(max(0, img_y), height - 1)
            
            # 获取像素颜色
            pixel_color = img[img_y, img_x]
            
            # 找到最接近的聚类中心
            distances = np.sum((centers - pixel_color) ** 2, axis=1)
            nearest_center_idx = np.argmin(distances)
            center_color = centers[nearest_center_idx]
            
            # 获取对应的z值
            z_value = get_z_value(center_color)
            zz[i, j] = z_value
            
            # 添加到数据列表
            if z_value > 0:  # 只添加有效值
                data.append({
                    x_label: xx[i, j],
                    y_label: yy[i, j],
                    z_label: z_value
                })
    
    # 创建3D图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xx, yy, zz, cmap='viridis', edgecolor='none')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()
    
    # 返回DataFrame
    return pd.DataFrame(data)

def main():
    print("请输入图像文件路径:")
    image_path = input().strip()
    
    if not image_path:  # 如果用户没有输入，使用默认路径
        image_path = "adsorption_figure.png"
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 文件 '{image_path}' 不存在")
        return
    
    try:
        # 提取数据
        cd_data, as_data, merged_data = extract_data_from_figure(image_path)
        
        # 保存文件
        cd_file = "cd_adsorption_data.csv"
        as_file = "as_adsorption_data.csv"
        merged_file = "cd_as_adsorption_data.csv"
        
        cd_data.to_csv(cd_file, index=False)
        as_data.to_csv(as_file, index=False)
        merged_data.to_csv(merged_file, index=False)
        
        print(f"\nCd(II)吸附数据已保存至: {cd_file}")
        print(f"As(III)吸附数据已保存至: {as_file}")
        print(f"合并数据已保存至: {merged_file}")
        
        # 显示部分结果
        print("\nCd(II)吸附数据样本:")
        print(cd_data.head())
        
        print("\nAs(III)吸附数据样本:")
        print(as_data.head())
        
        print("\n合并数据样本:")
        print(merged_data.head())
        
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")

if __name__ == "__main__":
    main()