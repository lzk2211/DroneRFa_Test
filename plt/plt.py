import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'

# Data for the plot
# 256
scenarios = ['D00', 'D01', 'D10']
resnet18_256_accuracies = [94.065, 85.47333, 93.8389430]  # Example accuracies for ResNet18
lw_resnet18_256_accuracies = [92.024, 94.9023, 91.845260]  # Example accuracies for 1D-LW-ResNet18
resnet18_1024_accuracies = [99.1875, 98.625, 98.25]  # Example accuracies for ResNet18
lw_resnet18_1024_accuracies = [98.25, 98.125, 96.625]  # Example accuracies for 1D-LW-ResNet18


# Bar width
bar_width = 0.35
index = np.arange(len(scenarios))

figure ,axes =plt.subplots(1,2,figsize=(6,3), tight_layout=True)
# 设置字体
# plt.rc('font', family='Times New Roman')

# 绘制第一个子图（准确率）
axes[0].bar(index, resnet18_1024_accuracies, bar_width, label='ResNet18', alpha=0.7)
axes[0].bar(index + bar_width, lw_resnet18_1024_accuracies, bar_width, label='1D-LW-ResNet18', alpha=0.7)

axes[0].set_xlabel('Distance Scenarios', fontsize=10)
axes[0].set_ylabel('Accuracy (%)', fontsize=10)
axes[0].set_title('The time and frequency scales are 1024', fontsize=12)
axes[0].set_xticks(index + bar_width / 2)
axes[0].set_xticklabels(scenarios)
# axes[0].legend()
axes[0].grid()

# 绘制第二个子图（损失）
axes[1].bar(index, resnet18_256_accuracies, bar_width, label='ResNet18', alpha=0.7)
axes[1].bar(index + bar_width, lw_resnet18_256_accuracies, bar_width, label='1D-LW-ResNet18', alpha=0.7)

axes[1].set_xlabel('Distance Scenarios', fontsize=10)
axes[1].set_title('The time and frequency scales are 256', fontsize=12)
axes[1].set_xticks(index + bar_width / 2)
axes[1].set_xticklabels(scenarios)
# axes[1].legend()
axes[1].grid()

# Ensure both subplots have the same y-axis limits
# y_limit = max(max(resnet18_1024_accuracies + lw_resnet18_1024_accuracies),
#               max(resnet18_256_accuracies + lw_resnet18_256_accuracies)) + 5
axes[0].set_ylim(80, 100)
axes[1].set_ylim(80, 100)

handles, labels = axes[0].get_legend_handles_labels()
figure.legend(handles, labels, loc='center right', bbox_to_anchor=(1.25, 0.5))

plt.subplots_adjust(left=0.2, right=0.85, top=0.95, bottom=0.5, wspace=0.1)
# 调整布局
plt.tight_layout()
plt.savefig('./acc.eps',bbox_inches='tight')  # 保存整个画布

# 显示图像
plt.show()

# 纵坐标最好scale相同，非定序数据不能折线图，条形统计图