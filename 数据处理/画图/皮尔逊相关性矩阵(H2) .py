import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['axes.unicode_minus'] = False 
# ── 数据 ──────────────────────────────────────────────
data_all = {
    'C0W1':  {'Cd-F1':[1.0275,2.205,3.48],  'Cd-RO':[23.8,13.15,11.35], 'Cd-R':[0.16,0.23,0.12], 'Cd-ST':[0.45,0.76,0.52], 'Cd-L':[0.33,0.31,0.21], },
    'C0W2':  {'Cd-F1':[2.325,0.78,0.9975],  'Cd-RO':[33.2,18.55,19.45], 'Cd-R':[0.87,0.24,0.17], 'Cd-ST':[3.03,1.86,1.13], 'Cd-L':[0.76,0.23,0.25], },
    'C0W3':  {'Cd-F1':[2.715,0.72,3.088],   'Cd-RO':[32.5,23.35,48.6],  'Cd-R':[1.63,0.77,1.62], 'Cd-ST':[5.89,2.39,7.52], 'Cd-L':[1.33,0,0.46]  },
    'C3W1':  {'Cd-F1':[1.23,1.95,0.9375],   'Cd-RO':[8.8,12.2,8.85],   'Cd-R':[0.14,0.21,0.09], 'Cd-ST':[0.59,0.96,0.7],  'Cd-L':[0.34,0.32,0.5]  },
    'C6W1':  {'Cd-F1':[1.95,1.3725,1.845],  'Cd-RO':[10.9,10.85,11.2], 'Cd-R':[0.08,0.27,0.15], 'Cd-ST':[0.75,1.27,1.33], 'Cd-L':[0.06,0.41,0.31]},
    'C6W2':  {'Cd-F1':[1.23,1.35,1.528],    'Cd-RO':[21,18.25,14.05],  'Cd-R':[0.36,0.19,0.12], 'Cd-ST':[1.76,1.45,1.45], 'Cd-L':[0.56,0.46,0.37] },
    'C6W3':  {'Cd-F1':[2.5275,1.4025,2.6],  'Cd-RO':[30.15,33.2,26.9], 'Cd-R':[0.661,0.62,0.55],'Cd-ST':[2.66,3.19,2.65], 'Cd-L':[0.02,1.17,0.93]},
}

cols = ['Cd-F1','Cd-RO','Cd-R','Cd-ST','Cd-L']

colors = ['#2166ac','#4393c3','#92c5de','#d1e5f0','#f7f7f7',
          '#fddbc7','#f4a582','#d6604d','#b2182b']
cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
plt.rcParams['font.family'] = 'Times New Roman'

FONTSIZE = 18

def plot_corr(ax, title, df_or_dict):
    df = pd.DataFrame(df_or_dict)[cols] if isinstance(df_or_dict, dict) else df_or_dict
    corr = df.corr(method='pearson').round(2)
    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap=cmap, vmin=-1, vmax=1,
        linewidths=0.5, linecolor='white', square=True,
        annot_kws={'size': FONTSIZE, 'fontfamily': 'Times New Roman', 'fontweight': 'bold'},
        cbar=True,
        cbar_kws={'shrink': 0.85, 'pad': 0.02},
        ax=ax
    )
    for text in ax.texts:
        text.set_color('black')

    # 色条字体
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1.0, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])
    cbar.ax.tick_params(length=6, width=2)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    ax.set_title(title, fontfamily='Times New Roman', fontsize=FONTSIZE, fontweight='bold', pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), fontfamily='Times New Roman', fontsize=FONTSIZE, rotation=0, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), fontfamily='Times New Roman', fontsize=FONTSIZE, rotation=0)
    ax.tick_params(length=0)

def merge_groups(keys):
    return pd.concat([pd.DataFrame(data_all[k])[cols] for k in keys], ignore_index=True)


# ══════════════════════════════════════════════════════
# 图1：C0W1 / C0W2 / C0W3
# ══════════════════════════════════════════════════════
# C0 合并所有W，C6 合并所有W，画一张图两个子图
c0_df = merge_groups(['C0W1', 'C0W2', 'C0W3'])
c6_df = merge_groups(['C6W1', 'C6W2', 'C6W3'])

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plot_corr(axes[0], 'C0', c0_df)
plot_corr(axes[1], 'C6', c6_df)

plt.tight_layout(rect=[0.02, 0, 1, 0.95])
plt.subplots_adjust(left=0.15, bottom=0.2)
plt.savefig('C0_C6.png', dpi=600)
plt.show()

# ══════════════════════════════════════════════════════
# 图3：W1 / W2 / W3
# ══════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(24, 7))
# fig3.suptitle('W1 / W2 / W3 (merged)', fontfamily='Times New Roman', fontsize=FONTSIZE+2, fontweight='bold')
for ax, (df, title) in zip(axes3, [
    (merge_groups(['C0W1','C3W1','C6W1']), 'W1'),
    (merge_groups(['C0W2','C6W2']),        'W2'),
    (merge_groups(['C0W3','C6W3']),        'W3'),
]):
    plot_corr(ax, title, df)
plt.tight_layout(rect=[0.02, 0, 1, 0.95])

plt.savefig('W123_merged.png', dpi=600)
plt.show()