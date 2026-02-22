import math
import argparse
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pywt
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ProcessingConfig:
    """数据处理配置参数"""
    # 数据路径配置
    data_dir: str = 'data'
    output_dir: str = 'data/processed'

    # 固定的处理参数（写死）
    windows: List[int] = None
    columns_to_plot: List[str] = None
    wavelet_type: str = 'haar'

    # 可视化选项
    visualize: bool = False

    def __post_init__(self):
        # 写死的参数
        self.windows = [8, 16, 32, 64, 128]
        self.columns_to_plot = ['开盘价(元)', '最高价(元)', '最低价(元)', '收盘价(元)']


class ArgumentsParser:
    """命令行参数解析器"""

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(
            description='金融时间序列小波变换数据处理工具',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # 数据路径参数
        parser.add_argument(
            '--data_dir',
            type=str,
            default='data',
            help='输入Excel数据文件目录路径'
        )

        parser.add_argument(
            '--output_dir',
            type=str,
            default='data/processed',
            help='输出numpy矩阵文件保存目录'
        )

        parser.add_argument(
            '--wavelet',
            type=str,
            default='haar',
            choices=['haar', 'db4', 'bior1.3', 'coif3', 'sym5'],
            help='小波变换类型'
        )

        # 可视化选项
        parser.add_argument(
            '--visualize',
            action='store_true',
            help='是否生成可视化图表'
        )

        # 调试选项
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='显示详细处理信息'
        )

        return parser.parse_args()


def run_dwt_transformation(data, wavelet='db4', level=4, visualize=False):
    """
    离散小波变换分析和可视化

    Parameters:
    - data: 金融时间序列数据
    - wavelet: 小波类型
    - level: 分解层数
    - visualize: 是否可视化
    """
    print("time series length:", len(data))

    # 提取价格序列
    if hasattr(data, 'values'):
        prices = data.values
    else:
        prices = data

    original_signal = prices
    N = len(original_signal)

    print(f"\nPerforming DWT analysis with {wavelet} wavelet, level={level}")

    # 1. 执行DWT分解
    coeffs = pywt.wavedec(original_signal, wavelet, level=level)

    # 创建系数可视化矩阵
    coeff_matrix = []

    # 为可视化创建系数矩阵
    for i, coeff in enumerate(coeffs):
        # 调整系数长度用于可视化
        resized_coeff = np.zeros(N)
        # 近似系数（CA）应该重复 N/len(coeff) 次
        repeat_factor = N // len(coeff)
        for j in range(len(coeff)):
            start_pos = j * repeat_factor
            end_pos = (j + 1) * repeat_factor if j < len(coeff) - 1 else N
            resized_coeff[start_pos:end_pos] = coeff[j]
        coeff_matrix.append(resized_coeff)

    coeff_matrix = np.array(coeff_matrix)

    if visualize:
        # 2. 创建可视化 - 单独绘制小波系数矩阵图
        fig, ax = plt.subplots(figsize=(14, 8))

        # 显示系数矩阵
        im = ax.pcolormesh(
            np.arange(N + 1),  # x坐标边界
            np.arange(len(coeffs) + 1),  # y坐标边界
            coeff_matrix,
            cmap='RdBu_r',
            shading='flat'
        )

        # 移除所有坐标轴元素
        ax.axis('off')  # 完全关闭坐标轴

        plt.tight_layout()
        plt.savefig(f'dwt_coefficients_only.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()  # 关闭图形以释放内存

    return coeff_matrix


class FinancialDataProcessor:
    """金融数据处理器"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.dfs = {}
        self.tickers = []

    def load_data(self):
        """加载Excel数据"""
        import pandas as pd
        import os

        print(f"正在从 {self.config.data_dir} 加载数据...")

        # 读取Excel数据
        self.dfs = {
            f"{file.rstrip('.xlsx')}": pd.read_excel(
                f'{self.config.data_dir}/{file}',
                engine='openpyxl',
                dtype_backend='numpy_nullable'
            )
            for file in os.listdir(self.config.data_dir)
            if file.endswith('.xlsx')
        }

        self.tickers = list(self.dfs.keys())
        print(f"加载完成，共找到 {len(self.tickers)} 个股票数据: {self.tickers}")

    def process_ticker(self, ticker: str):
        """处理单个股票数据"""
        print(f"正在处理股票: {ticker}")

        results_dict = {}

        for window in self.config.windows:
            print(f"  正在处理窗口大小: {window}")
            results = []

            # 计算分解层数
            level = int(math.log2(window))

            for i in range(len(self.dfs[ticker]) - window + 1):
                # 获取窗口内的所有价格数据
                all_prices = [row for _, row in self.dfs[ticker].iterrows()][i:i + window]
                prices = [[row[column] for row in all_prices] for column in self.config.columns_to_plot]

                # 对每个价格序列执行小波变换
                mats = np.array([
                    run_dwt_transformation(
                        price,
                        wavelet=self.config.wavelet_type,
                        level=level,
                        visualize=False
                    ) for price in prices
                ])

                results.append(mats)

            results_dict[window] = np.array(results)
            print(f"    窗口 {window} 处理完成，结果形状: {results_dict[window].shape}")

        return results_dict

    def save_results(self, ticker: str, results_dict: dict):
        """保存处理结果"""
        import os

        # 创建输出目录
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
            print(f"创建输出目录: {self.config.output_dir}")

        # 保存每个窗口的结果
        for window, results in results_dict.items():
            output_path = f'{self.config.output_dir}/{ticker}_{window}.npy'
            np.save(output_path, results)
            print(f"    已保存: {output_path}")


def main():
    """主执行函数"""
    # 解析命令行参数
    args = ArgumentsParser.parse_arguments()

    # 创建配置对象（windows和columns已写死）
    config = ProcessingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        wavelet_type=args.wavelet,
        visualize=args.visualize
    )

    # 设置matplotlib后端
    import matplotlib
    matplotlib.use('Agg')

    # 初始化处理器
    processor = FinancialDataProcessor(config)

    # 加载数据
    processor.load_data()

    # 处理每个股票
    for ticker in processor.tickers:
        try:
            results = processor.process_ticker(ticker)
            processor.save_results(ticker, results)
            print(f"✓ 股票 {ticker} 处理完成\n")
        except Exception as e:
            print(f"✗ 处理股票 {ticker} 时出错: {str(e)}\n")
            if args.verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
