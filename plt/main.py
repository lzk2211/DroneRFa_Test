import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def smooth_acc(csv_path, weight=0.6):
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    indexs = []
    scalars = []
    for point in scalar:
        scalars.append(point*100)

        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val*100)
        last = smoothed_val

    for index in data['Step']:
        indexs.append(index + 1)

    return indexs, scalars, smoothed

def smooth_loss(csv_path, weight=0.6):
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    indexs = []
    scalars = []
    for point in scalar:
        scalars.append(point)

        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    for index in data['Step']:
        indexs.append(index + 1)

    return indexs, scalars, smoothed

def plot_acc_graph():
    file1 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_vector_group_D00_1024_1024_dim2_bs128_lr0.0001_runs.csv'
    file1_1 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_vector_group_D00_1024_1024_dim2_bs128_lr0.0001_runs_loss.csv'

    file2 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_vector_group_D00_256_256_dim2_bs128_lr0.0001_runs.csv'
    file2_1 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_vector_group_D00_256_256_dim2_bs128_lr0.0001_runs_loss.csv'

    file3 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_pre_D00_1024_1024_bs128_lr0.0001_runs.csv'
    file3_1 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_pre_D00_1024_1024_bs128_lr0.0001_runs_loss.csv'

    file4 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_pre_D00_256_256_bs128_lr0.0001_runs.csv'
    file4_1 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_pre_D00_256_256_bs128_lr0.0001_runs_loss.csv'

    index1, scalar1, smoothed1 = smooth_loss(file3_1)
    index2, scalar2, smoothed2 = smooth_loss(file4_1)
    index3, scalar3, smoothed3 = smooth_loss(file1_1)
    index4, scalar4, smoothed4 = smooth_loss(file2_1)

    # 绘制原始数据和平滑数据
    label1 = 'D00_1024_1024'
    label2 = 'D00_256_256'
    # plt.figure(figsize=(5, 5))
    figure ,axes =plt.subplots(1,2,figsize=(10,5))
    # 设置字体
    plt.rc('font', family='Times New Roman')

    # 绘制第一个子图（准确率）
    axes[0].plot(index1, scalar1, color='#00BFFF', label='D00_1024_1024', linewidth=1)
    # axes[0].plot(index1, smoothed1, label='D00_1024_1024', color='#00BFFF', linewidth=3.0, alpha=1)
    axes[0].plot(index2, scalar2, color='red', label='D00_256_256', linewidth=1)
    # axes[0].plot(index2, smoothed2, label='D00_256_256', color='red', linewidth=3.0, alpha=1)

    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid()
    # axes[0].set_title('ResNet18')
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))  # 只显示整数刻度


    # 绘制第二个子图（损失）
    axes[1].plot(index3, scalar3, color='#00BFFF', label='D00_1024_1024', linewidth=1)
    # axes[1].plot(index3, smoothed3, label='D00_1024_1024', color='#00BFFF', linewidth=3.0, alpha=1)
    axes[1].plot(index4, scalar4, color='red', label='D00_256_256', linewidth=1)
    # axes[1].plot(index4, smoothed4, label='D00_256_256', color='red', linewidth=3.0, alpha=1)

    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid()
    # axes[1].set_title('Loss Comparison')\
    # axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))  # 只显示整数刻度


    # 调整布局
    plt.tight_layout()

    # 显示图像
    plt.show()



if __name__=='__main__':
    plot_acc_graph()
