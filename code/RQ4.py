import matplotlib.pyplot as plt
import numpy as np

# 使用默认字体，避免 Arial 报错
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# 原始 reduction values (用于标签)
reduction_vals = [4, 8, 16, 32, 64]
# 等间距的 x 位置（0 到 4）
x = np.arange(len(reduction_vals))  # [0, 1, 2, 3, 4]

# 数据
y_blue_n = np.array([0.7898, 0.7924, 0.7875, 0.8104, 0.7368])
y_red_n = np.array([0.8015, 0.7908, 0.7907, 0.8164, 0.7386])

y_blue_s = np.array([0.7555, 0.7457, 0.7555, 0.7630, 0.6910])
y_red_s = np.array([0.7814, 0.7812, 0.7903, 0.7924, 0.7348])

y_blue_p = np.array([0.7361, 0.7366, 0.7494, 0.7608, 0.6827])
y_red_p = np.array([0.7854, 0.7771, 0.7918, 0.7921, 0.7383])

y_blue_sn = np.array([0.6808, 0.5833, 0.7356, 0.6086, 0.6593])
y_red_sn = np.array([0.6846, 0.5898, 0.7499, 0.6111, 0.6758])

label_fontsize = 20  # 设置字体大小

# 创建图形
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 7), sharex=True)  # 增加宽度以便为图例留出空间

yticks = [0.70, 0.85]

yticks_ax4 = [0.55, 0.78]

# 定义颜色
color_blue = '#2E75B6'
color_red = '#C55A11'

# 子图 1: As
ax1.plot(x, y_blue_s, color=color_blue, linestyle='--', marker='*', markersize=15, linewidth=1.5)
ax1.plot(x, y_red_s, color=color_red, linestyle='--', marker='*', markersize=15, linewidth=1.5)
ax1.set_ylabel(r'$\mathcal{A}_s$', rotation=0, labelpad=25, va='center', fontsize=label_fontsize, fontweight='bold')
ax1.set_ylim(0.66, 0.88)
ax1.set_yticks(yticks)
ax1.tick_params(axis='y', labelsize=12)
# ax1.grid(True, alpha=0.3)

# 子图 2: Ap
ax2.plot(x, y_blue_p, color=color_blue, linestyle='--', marker='*', markersize=15, linewidth=1.5)
ax2.plot(x, y_red_p, color=color_red, linestyle='--', marker='*', markersize=15, linewidth=1.5)
ax2.set_ylabel(r'$\mathcal{A}_p$', rotation=0, labelpad=25, va='center', fontsize=label_fontsize, fontweight='bold')
ax2.set_ylim(0.66, 0.88)
ax2.set_yticks(yticks)
ax2.tick_params(axis='y', labelsize=12)
# ax2.grid(True, alpha=0.3)

# 子图 3: An
ax3.plot(x, y_blue_n, color=color_blue, linestyle='--', marker='*', markersize=15, linewidth=1.5)
ax3.plot(x, y_red_n, color=color_red, linestyle='--', marker='*', markersize=15, linewidth=1.5)
ax3.set_ylabel(r'$\mathcal{A}_n$', rotation=0, labelpad=25, va='center', fontsize=label_fontsize, fontweight='bold')
ax3.set_ylim(0.66, 0.88)
ax3.set_yticks(yticks)
ax3.tick_params(axis='y', labelsize=12)
# ax3.grid(True, alpha=0.3)

# 设置等间距 x 轴
ax4.plot(x, y_blue_sn, color=color_blue, linestyle='--', marker='*', markersize=15, linewidth=1.5)
ax4.plot(x, y_red_sn, color=color_red, linestyle='--', marker='*', markersize=15, linewidth=1.5)
ax4.set_ylabel(r'$\mathcal{B}_{sn}$', rotation=0, labelpad=25, va='center', fontsize=label_fontsize, fontweight='bold')
ax4.set_ylim(0.56, 0.76)
ax4.set_yticks(yticks_ax4)
ax4.tick_params(axis='y', labelsize=12)

ax4.set_xticks(x)
ax4.set_xticklabels([str(v) for v in reduction_vals])
ax4.set_xlabel('reduction', fontsize=label_fontsize)
ax4.tick_params(axis='x', labelsize=12)
ax4.tick_params(axis='y', labelsize=12)

# 图例：MaF1 / MiF1
handles = [
    plt.Line2D([], [], color='#2E75B6', linestyle='--', marker='*', markersize=15, linewidth=1.5),
    plt.Line2D([], [], color='#C55A11', linestyle='--', marker='*', markersize=15, linewidth=1.5)
]
labels = ['MaF1', 'MiF1']

fig.legend(
    handles, labels,
    loc='center left',
    bbox_to_anchor=(0.86, 0.5),
    frameon=False,
    handlelength=2,
    fontsize=15
)

plt.tight_layout()
plt.subplots_adjust(right=0.86)  # 留出刚好容纳图例的空间
plt.savefig('reduction.pdf', format='pdf', bbox_inches='tight')
plt.show()