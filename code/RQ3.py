import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 数据
mif1_data = {
    r'$\mathcal{A}_s$': [0.75, 0.71, 0.70, 0.25, 0.76],
    r'$\mathcal{A}_p$': [0.74, 0.75, 0.70, 0.31, 0.77],
    r'$\mathcal{A}_n$': [0.78, 0.71, 0.76, 0.37, 0.78]
}

maf1_data = {
    r'$\mathcal{A}_s$': [0.79, 0.72, 0.72, 0.46, 0.78],
    r'$\mathcal{A}_p$': [0.79, 0.78, 0.74, 0.54, 0.80],
    r'$\mathcal{A}_n$': [0.79, 0.72, 0.76, 0.54, 0.79]
}

labels = [r'$\mathcal{A}_s$', r'$\mathcal{A}_p$', r'$\mathcal{A}_n$']
hatches = ['++', '..', '//', '\\\\', 'oo']
colors = ['lightgray'] * len(hatches)

plt.rcParams['mathtext.fontset'] = 'dejavusans'

# 创建图形，留出右侧空间给图例
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

x = np.arange(len(labels))
width = 0.14
label_fontsize = 18  # 设置字体大小
yticks = [0, 0.9]

# MiF1
for i, group in enumerate(labels):
    values = mif1_data[group]
    for j, val in enumerate(values):
        offset = x[i] - width * (len(values)//2 - j)
        ax1.bar(offset, val, width,
                hatch=hatches[j], color=colors[j],
                edgecolor='black', linewidth=0.6, zorder=3)
        ax1.text(offset, val + 0.012, f'{val:.2f}', 
                 ha='center', va='bottom', fontsize=8, zorder=4)

ax1.set_ylabel(r'$MiF1$', rotation=0, labelpad=25, va='center', fontsize=label_fontsize, fontweight='bold')

ax1.set_ylim(0, 0.85)
ax1.set_yticks(yticks)
ax1.tick_params(axis='y', labelsize=16)
# ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# MaF1
for i, group in enumerate(labels):
    values = maf1_data[group]
    for j, val in enumerate(values):
        offset = x[i] - width * (len(values)//2 - j)
        ax2.bar(offset, val, width,
                hatch=hatches[j], color=colors[j],
                edgecolor='black', linewidth=0.6, zorder=3)
        ax2.text(offset, val + 0.012, f'{val:.2f}', 
                 ha='center', va='bottom', fontsize=8, zorder=4)

ax2.set_ylabel(r'$MaF1$', rotation=0, labelpad=25, va='center', fontsize=label_fontsize, fontweight='bold')

ax2.set_ylim(0, 0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_yticks(yticks)
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)
# ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# === 图例：放在整图右侧，垂直居中 ===
legend_elements = [
    Patch(facecolor='lightgray', edgecolor='black', hatch=hatches[0], label='M'),
    Patch(facecolor='lightgray', edgecolor='black', hatch=hatches[1], label='M1'),
    Patch(facecolor='lightgray', edgecolor='black', hatch=hatches[2], label='M2'),
    Patch(facecolor='lightgray', edgecolor='black', hatch=hatches[3], label='M3'),
    Patch(facecolor='lightgray', edgecolor='black', hatch=hatches[4], label='M4'),
]

# 关键：使用 fig.legend 并定位到右侧
fig.legend(handles=legend_elements, 
           loc='center left', 
           bbox_to_anchor=(0.85, 0.5),   # (x=0.98, y=0.5) → 右侧，垂直居中
           fontsize=9)

# 调整子图间距，为右侧图例留空间
fig.tight_layout(rect=[0, 0, 0.85, 0.96])  # 右边界留到 0.85，图例在 0.85～1.0

# 标题
# fig.suptitle('Figure 5: Contributions of the multimodal data', fontsize=12)

# 保存
plt.savefig("figure5.pdf", format='pdf', bbox_inches='tight')
# plt.show()  # 如需显示可取消注释