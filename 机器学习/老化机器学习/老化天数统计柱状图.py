import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set English font to Times New Roman with bold weight
rcParams['font.family'] = 'Times New Roman'
rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'

# Climate data
regions = ['NT', 'NC', 'CC', 'EC', 'SC', 'SW']
dry_wet_days = [2, 3, 4, 5, 10, 10]
freeze_thaw_days = [160, 109, 43, 2, 0, 10]

# Set colors as specified
dry_wet_color = '#ffb7b8'  # Red color for dry-wet cycle (BR)
freeze_thaw_color = '#b7dbff'  # Blue color for freeze-thaw cycle (OS)

# Use same colors for all bars
dry_wet_bar_colors = [dry_wet_color] * len(dry_wet_days)
freeze_thaw_bar_colors = [freeze_thaw_color] * len(freeze_thaw_days)

# Create figure and axis with dual y-axis
fig, ax1 = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('#f8f9fa')

# Position for bars
x_pos = np.arange(len(regions))
width = 0.35

# Plot dry-wet cycle bars on left y-axis
bars1 = ax1.bar(x_pos - width/2, dry_wet_days, width, color=dry_wet_bar_colors, 
                edgecolor='black', linewidth=1, alpha=0.9, label='DW')

# # Add data labels for dry-wet bars
# for i, (bar, value) in enumerate(zip(bars1, dry_wet_days)):
#     height = bar.get_height()
#     ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
#              f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Set left y-axis properties
# ax1.set_xlabel('Geographical Regions', fontsize=14, fontweight='bold')
ax1.set_ylabel('干湿循环天数', fontsize=22, fontweight='bold', color='#ffb7b8')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(regions, fontsize=22, fontweight='bold', rotation=0, ha='center')
ax1.set_ylim(0, max(dry_wet_days) * 1.2)
ax1.tick_params(axis='y', labelcolor='#ffb7b8', labelsize=20)
ax1.tick_params(axis='x', labelsize=20)
# Make tick labels bold
for label in ax1.get_xticklabels():
    label.set_fontweight('bold')
for label in ax1.get_yticklabels():
    label.set_fontweight('bold')
# ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_facecolor('#ffffff')
ax1.spines['top'].set_visible(False)
# Create right y-axis for freeze-thaw cycle
ax2 = ax1.twinx()

# Plot freeze-thaw cycle bars on right y-axis
bars2 = ax2.bar(x_pos + width/2, freeze_thaw_days, width, color=freeze_thaw_bar_colors, 
                edgecolor='black', linewidth=1, alpha=0.9, label='FT')

# # Add data labels for freeze-thaw bars
# for i, (bar, value) in enumerate(zip(bars2, freeze_thaw_days)):
#     height = bar.get_height()
#     ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
#              f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Set right y-axis properties
ax2.set_ylabel('冻融循环天数', fontsize=22, fontweight='bold', color='#b7dbff')
ax2.set_ylim(0, max(freeze_thaw_days) * 1.1)
ax2.tick_params(axis='y', labelcolor='#b7dbff', labelsize=20)
ax2.tick_params(top=False, labeltop=False)
ax2.spines['top'].set_visible(False)
# Make right y-axis tick labels bold
for label in ax2.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(20)

# Add legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=20, frameon=True)
# Make legend text bold
for text in legend.get_texts():
    text.set_fontweight('bold')

# # Add borders
# for spine in ax1.spines.values():
#     spine.set_edgecolor('#2c3e50')
#     spine.set_linewidth(0)

# Adjust layout
plt.tight_layout()


plt.savefig('气候对应老化天数统计图.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Close the figure to prevent display
plt.close()

# Print data table in English
print("\n" + "="*70)
print("Climate Characteristics Data of Six Geographical Regions in China")
print("="*70)
print(f"{'Region':<15} {'Dry-Wet Cycle (days)':<20} {'Freeze-Thaw Cycle (days)':<20}")
print("-"*70)
for i, region in enumerate(regions):
    print(f"{region:<15} {dry_wet_days[i]:<20} {freeze_thaw_days[i]:<20}")
print("="*70)

print("\nFigures saved as:")
print("- China_Climate_Characteristics_Dual_Axis.png")
print("- China_Climate_Characteristics_Dual_Axis.pdf")

# Additional function to create a comparison table
def print_detailed_analysis():
    """Print detailed analysis of the climate data"""
    print("\n" + "="*70)
    print("DETAILED CLIMATE ANALYSIS")
    print("="*70)
    
    print("\nDry-Wet Cycle Analysis:")
    print(f"• Minimum: {min(dry_wet_days)} days (Northeast)")
    print(f"• Maximum: {max(dry_wet_days)} days (South China, Southwest)")
    print(f"• Average: {np.mean(dry_wet_days):.1f} days")
    print(f"• Standard Deviation: {np.std(dry_wet_days):.1f} days")
    
    print("\nFreeze-Thaw Cycle Analysis:")
    print(f"• Minimum: {min(freeze_thaw_days)} days (South China)")
    print(f"• Maximum: {max(freeze_thaw_days)} days (Northeast)")
    print(f"• Average: {np.mean(freeze_thaw_days):.1f} days")
    print(f"• Standard Deviation: {np.std(freeze_thaw_days):.1f} days")
    
    print("\nRegional Climate Patterns:")
    for i, region in enumerate(regions):
        dry_wet = dry_wet_days[i]
        freeze_thaw = freeze_thaw_days[i]
        
        if freeze_thaw > 100:
            climate_type = "Cold Climate"
        elif freeze_thaw == 0:
            climate_type = "Tropical Climate"
        elif dry_wet >= 10:
            climate_type = "Humid Climate"
        else:
            climate_type = "Temperate Climate"
            
        print(f"• {region}: {climate_type} (DW: {dry_wet}d, FT: {freeze_thaw}d)")

# Run detailed analysis
print_detailed_analysis()