import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def smooth(csv_path, weight=0.6):
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

    # save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
    # save.to_csv('smooth_'+csv_path)

def plot_acc_graph():
    file1 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_vector_group_D00_1024_1024_dim2_bs128_lr0.0001_runs.csv'
    file1_1 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_vector_group_D00_1024_1024_dim2_bs128_lr0.0001_runs_loss.csv'

    file2 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_vector_group_D00_256_256_dim2_bs128_lr0.0001_runs.csv'
    file2_1 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_vector_group_D00_256_256_dim2_bs128_lr0.0001_runs_loss.csv'

    file3 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_pre_D00_1024_1024_bs128_lr0.0001_runs.csv'
    file3_1 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_pre_D00_1024_1024_bs128_lr0.0001_runs_loss.csv'

    file4 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_pre_D00_256_256_bs128_lr0.0001_runs.csv'
    file4_1 = r'C:\Users\BUPT\Desktop\无人机论文\lzk\csv\resnet18_pre_D00_256_256_bs128_lr0.0001_runs_loss.csv'

    index1, scalar1, smoothed1 = smooth(file1_1)
    index2, scalar2, smoothed2 = smooth(file2_1)

    # 绘制原始数据和平滑数据
    label1 = 'D00_1024_1024'
    label2 = 'D00_256_256'
    plt.figure(figsize=(5, 5))
    plt.plot(index1, scalar1, color='#00BFFF', linewidth=1)
    plt.plot(index1, smoothed1, label=label1, color='#00BFFF', linewidth=3.0, alpha=1)
    plt.plot(index2, scalar2, color='red', linewidth=1)
    plt.plot(index2, smoothed2, label=label2, color='red', linewidth=3.0, alpha=1)
    plt.rc('font',family='Times New Roman')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    # plt.title('1D-LW-ResNet18 model ')
    plt.legend()
    plt.grid()
    # plt.savefig('smoothed_plot.png')  # 保存图像
    plt.show()  # 显示图像


if __name__=='__main__':
    plot_acc_graph()
