import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_decision_tree_classification(X_scaled, y, X_original, title, output_path='decision_tree_classification.png'):
   plt.rcParams['font.family'] = 'SimSun'

   colors = [
       '#FF0000', '#00A86B', '#0000FF', '#FFA500', 
       '#800080', '#00FFFF', '#FFC0CB', '#008000'
   ]

   clf = DecisionTreeClassifier(
    max_depth=3,           # 降低深度
    min_samples_split=2,  # 增加分裂所需的最小样本数
    min_samples_leaf=2,    # 增加叶子节点所需的最小样本数
    random_state=42
)
   clf.fit(X_scaled, y)
   
   fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
   
   x_min, x_max = X_original['As'].min() - 0.5, X_original['As'].max() + 0.5
   y_min, y_max = X_original['Cd'].min() - 0.5, X_original['Cd'].max() + 0.5
   
   xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                        np.linspace(y_min, y_max, 300))
   
   grid_points = np.c_[xx.ravel(), yy.ravel()]
   grid_points_scaled = StandardScaler().fit(X_original).transform(grid_points)
   
   Z = clf.predict(grid_points_scaled)
   Z = Z.reshape(xx.shape)
   
   levels = list(range(1, 9))
   color_dict = {cls: colors[int(cls)-1] for cls in levels}
   
   contours = ax.contourf(xx, yy, Z, alpha=0.3, levels=levels, 
                          colors=[color_dict.get(cls, '#FFFFFF') for cls in levels])

   ax.contour(xx, yy, Z, levels=levels, colors='black', linewidths=0.3, alpha=0.2)

   for treatment in range(1, 9):
       mask = y == treatment
       if mask.any():
           ax.scatter(X_original[mask]['As'], X_original[mask]['Cd'], 
                      color=colors[treatment-1], 
                      label=f'Treatment {treatment}', 
                      edgecolor='black')
   
   cbar = plt.colorbar(contours, ax=ax, ticks=levels, 
                       fraction=0.1, aspect=8, pad=0.08)
   
   cbar.set_ticklabels([f'Treatment {cls}' for cls in levels])
   for t, c in zip(cbar.ax.get_yticklabels(), colors):
       t.set_color(c)
   
   ax.set_xlabel('ln(RK) of As')
   ax.set_ylabel('ln(RK) of Cd')
   ax.set_title(title)
   ax.legend()
   
   ax.spines['left'].set_position('zero')
   ax.spines['bottom'].set_position('zero')
   ax.spines['right'].set_color('none')
   ax.spines['top'].set_color('none')

   plt.tight_layout()
   plt.savefig(output_path, bbox_inches='tight')
   plt.close(fig)

def decision_tree_analysis(file_path, feature_cols, target_col, output_path='decision_tree_classification.png'):
   df = pd.read_csv(file_path)
   
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(df[feature_cols])
   y = df[target_col]
   
   plot_decision_tree_classification(X_scaled, y, df[feature_cols], 
                           f'Decision Tree Classification', 
                           output_path)

decision_tree_analysis(r'C:\Users\AMDYE\OneDrive\python\数据处理\画图\water.csv',
                      feature_cols=['As', 'Cd'],
                      target_col='处理',
                      output_path='decision_tree_classification.png')