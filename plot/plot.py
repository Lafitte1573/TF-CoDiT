import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import pandas as pd
import openpyxl
import os

# 读取Excel数据
dfs = [
    pd.read_excel(f'../data/{file}', engine='openpyxl', dtype_backend='numpy_nullable')
    for file in os.listdir('../data')
]
for df in dfs:
    print(df.head())

# 定义需要绘制的列
columns_to_plot = ['开盘价(元)', '最高价(元)', '最低价(元)', '收盘价(元)', '结算价', '成交额(百万)', '成交量(股)', '持仓量']

# 创建图表和子图 - 修改为8行4列
fig, axes = plt.subplots(nrows=8, ncols=4, figsize=(20, 30))
# fig.suptitle('T.CFE 数据可视化', fontsize=16)

# 绘制每个子图 - 按行排列，每行4个子图
for i, column in enumerate(columns_to_plot):
    # 在第i行绘制4个相同的图表
    for j, df in enumerate(dfs):  # 每一行有4列
        ax = axes[i, j]
        ax.plot(df['日期'], df[column], linewidth=1)
        ax.set_title(f'表 - {df["代码"][0]}')
        # ax.set_xlabel('日期')
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)

        # 只在最后一行显示x轴刻度和标签
        if i == len(columns_to_plot) - 1:  # 最后一行
            ax.set_xlabel('日期')
            # 自动旋转x轴标签以避免重叠
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
        else:
            # 隐藏其他行的x轴刻度标签
            ax.tick_params(axis='x', labelbottom=False)

# 保存图像
plt.savefig('T_CFE_analysis.png', dpi=300, bbox_inches='tight')
print("图表已保存为 T_CFE_analysis.png")

# 显示图表（在非交互模式下可能不会显示）
# plt.show()
