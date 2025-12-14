import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pywt
import warnings

warnings.filterwarnings('ignore')


def fetch_financial_data(ticker="AAPL", period="6mo"):
    """Fetch financial time series data from yfinance"""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    print(f"Fetched {len(df)} data points for {ticker}")
    print(f"Time range: {df.index[0].date()} to {df.index[-1].date()}")
    return df


def dwt_analysis_and_visualization(data, ticker="AAPL", wavelet='db4', level=4):
    """
    Discrete Wavelet Transform analysis and visualization

    Parameters:
    - data: Financial time series data
    - ticker: Stock symbol for title
    - wavelet: Wavelet type for DWT
    - level: Decomposition level
    """
    print("time series length:", len(data))

    # Extract price series
    if hasattr(data, 'values'):
        prices = data.values
    else:
        prices = data

    # if len(prices.shape) > 1:
    #     prices = prices.flatten()

    original_signal = prices
    N = len(original_signal)

    print(f"\nPerforming DWT analysis with {wavelet} wavelet, level={level}")

    # 1. Perform DWT decomposition
    coeffs = pywt.wavedec(original_signal, wavelet, level=level)
    for key, value in enumerate(coeffs):
        print(f"Level {key}: {len(value)}")
        # print(f"Level {key}:", value)
    # print(coeffs)
    # exit()

    # Print decomposition structure
    print(f"Approximation coefficients (cA{level}) length: {len(coeffs[0])}")
    for i in range(1, level + 1):
        print(f"Detail coefficients (cD{i}) length: {len(coeffs[-i])}")

    # 2. Reconstruct signal
    reconstructed_signal = pywt.waverec(coeffs, wavelet)

    # Ensure same length (wavelet transform may cause boundary extension)
    min_len = min(len(original_signal), len(reconstructed_signal))
    original_trunc = original_signal[:min_len]
    reconstructed_trunc = reconstructed_signal[:min_len]

    # Calculate reconstruction error
    mse = np.mean((original_trunc - reconstructed_trunc) ** 2)
    print(f"\nReconstruction MSE: {mse:.10f}")
    print(f"Maximum absolute error: {np.max(np.abs(original_trunc - reconstructed_trunc)):.6f}")

    # 3. Create visualization
    fig = plt.figure(figsize=(16, 10))

    # Define grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1], width_ratios=[1, 1])

    # Subplot 1: Original signal
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(original_trunc, 'b-', linewidth=2)
    ax1.set_title(f'{ticker} - Original Price Series', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, min_len - 1])

    # Subplot 2: DWT coefficients (spectrum)
    ax2 = fig.add_subplot(gs[1, :])

    # Create coefficient visualization (spectrogram-like representation)
    coeff_matrix = []
    max_coeff_len = 0

    # Create a matrix of coefficients for visualization
    for i, coeff in enumerate(coeffs):
        print(f"Coeff {i}: {len(coeff)}")
        # Resize coefficients to same length for visualization
        resized_coeff = np.zeros(N)
        start_idx = 0
        # if i == 0:  # Approximation coefficients
        #     # Repeat approximation coefficients to match length
        #     repeat_factor = N // len(coeff)
        #     for j in range(len(coeff)):
        #         resized_coeff[j * repeat_factor:(j + 1) * repeat_factor] = coeff[j]
        # else:  # Detail coefficients
        #     # For details, we place them in their corresponding positions
        #     scale_factor = 2 ** i
        #     for j in range(len(coeff)):
        #         pos = j * scale_factor
        #         if pos < N:
        #             resized_coeff[pos:min(pos + scale_factor, N)] = coeff[j]
        #
        # coeff_matrix.append(resized_coeff)
        if i == 0:  # Approximation coefficients (cA_level)
            # 近似系数应该重复 N/len(coeff) 次
            repeat_factor = N // len(coeff)
            for j in range(len(coeff)):
                start_pos = j * repeat_factor
                end_pos = (j + 1) * repeat_factor if j < len(coeff) - 1 else N
                resized_coeff[start_pos:end_pos] = coeff[j]
        else:  # Detail coefficients (cD_i)
            # 细节系数也应该按照正确的比例重复
            repeat_factor = N // len(coeff)
            for j in range(len(coeff)):
                start_pos = j * repeat_factor
                end_pos = (j + 1) * repeat_factor if j < len(coeff) - 1 else N
                resized_coeff[start_pos:end_pos] = coeff[j]

        coeff_matrix.append(resized_coeff)

    coeff_matrix = np.array(coeff_matrix)
    # print(coeff_matrix)
    # exit()

    # Display the coefficient matrix
    # im = ax2.imshow(
    #     coeff_matrix,
    #     aspect='auto',
    #     cmap='RdBu_r',
    #     extent=[0, min_len - 1, len(coeffs), 0]
    # )
    # 替换原来的 imshow 代码
    im = ax2.pcolormesh(
        np.arange(min_len + 1),  # x坐标边界
        np.arange(len(coeffs) + 1),  # y坐标边界
        coeff_matrix,
        cmap='RdBu_r',
        shading='flat'
    )
    ax2.set_title('DWT Coefficient Matrix (Time-Frequency Representation)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Point', fontsize=10)
    ax2.set_ylabel('Decomposition Level', fontsize=10)
    ax2.set_yticks(range(1, len(coeffs) + 1))
    ax2.set_yticklabels([f'cA{level}'] + [f'cD{level - i + 1}' for i in range(1, len(coeffs))])
    plt.colorbar(im, ax=ax2, label='Coefficient Value')
    # plt.grid(True)

    # Subplot 3: Individual coefficient plots
    ax3 = fig.add_subplot(gs[2, 0])

    # Plot each coefficient set
    colors = plt.cm.viridis(np.linspace(0, 1, len(coeffs)))

    for i, (coeff, color) in enumerate(zip(coeffs, colors)):
        level_label = f'cA{level}' if i == 0 else f'cD{level - i + 1}'
        x_positions = np.linspace(0, min_len - 1, len(coeff))
        ax3.plot(x_positions, coeff, color=color, linewidth=1.5, label=level_label)

    ax3.set_title('DWT Coefficients by Level', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time Point', fontsize=9)
    ax3.set_ylabel('Coefficient Value', fontsize=9)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Reconstructed signal with error
    ax4 = fig.add_subplot(gs[2, 1])

    ax4.plot(original_trunc, 'b-', linewidth=2, alpha=0.7, label='Original')
    ax4.plot(reconstructed_trunc, 'r-', linewidth=2, alpha=0.7, label='Reconstructed')

    # Fill between to show error
    ax4.fill_between(
        range(min_len),
        original_trunc,
        reconstructed_trunc,
        alpha=0.2,
        color='gray',
        label='Error'
    )

    ax4.set_title(f'Signal Reconstruction (MSE: {mse:.2e})', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time Point', fontsize=9)
    ax4.set_ylabel('Price', fontsize=9)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'dwt.png')

    # 4. Return analysis results
    return {
        'original': original_trunc,
        'reconstructed': reconstructed_trunc,
        'coeffs': coeffs,
        'mse': mse,
        'wavelet': wavelet,
        'level': level
    }


def dwt_visualization(data, ticker="AAPL", wavelet='db4', level=4):
    """
    Discrete Wavelet Transform analysis and visualization

    Parameters:
    - data: Financial time series data
    - ticker: Stock symbol for title
    - wavelet: Wavelet type for DWT
    - level: Decomposition level
    """
    print("time series length:", len(data))

    # Extract price series
    if hasattr(data, 'values'):
        prices = data.values
    else:
        prices = data

    original_signal = prices
    N = len(original_signal)

    print(f"\nPerforming DWT analysis with {wavelet} wavelet, level={level}")

    # 1. Perform DWT decomposition
    coeffs = pywt.wavedec(original_signal, wavelet, level=level)
    for key, value in enumerate(coeffs):
        print(f"Level {key}: {len(value)}")

    # Print decomposition structure
    print(f"Approximation coefficients (cA{level}) length: {len(coeffs[0])}")
    for i in range(1, level + 1):
        print(f"Detail coefficients (cD{i}) length: {len(coeffs[-i])}")

    # 2. Create visualization - 单独绘制小波系数矩阵图
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create coefficient visualization (spectrogram-like representation)
    coeff_matrix = []

    # Create a matrix of coefficients for visualization
    for i, coeff in enumerate(coeffs):
        print(f"Coeff {i}: {len(coeff)}")
        # Resize coefficients to same length for visualization
        resized_coeff = np.zeros(N)
        # 近似系数（CA）应该重复 N/len(coeff) 次
        repeat_factor = N // len(coeff)
        for j in range(len(coeff)):
            start_pos = j * repeat_factor
            end_pos = (j + 1) * repeat_factor if j < len(coeff) - 1 else N
            resized_coeff[start_pos:end_pos] = coeff[j]
        # print(f"Level {i}, resized Coeff: {resized_coeff}")
        coeff_matrix.append(resized_coeff)

    coeff_matrix = np.array(coeff_matrix)

    # Display the coefficient matrix
    im = ax.pcolormesh(
        np.arange(N + 1),  # x坐标边界
        np.arange(len(coeffs) + 1),  # y坐标边界
        coeff_matrix,
        cmap='RdBu_r',
        shading='flat'
    )

    ax.set_title('DWT Coefficient Matrix (Time-Frequency Representation)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Point', fontsize=12)
    ax.set_ylabel('Decomposition Level', fontsize=12)
    ax.set_yticks(range(1, len(coeffs) + 1))
    ax.set_yticklabels([f'cA0'] + [f'cD{i}' for i in range(1, len(coeffs))])

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='Coefficient Value')
    cbar.set_label('Coefficient Value', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'dwt_coefficients_only.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存

    return coeff_matrix


def multi_resolution_analysis(data, ticker="AAPL", wavelet='db4', level=4):
    """
    Additional visualization: Multi-resolution analysis
    Shows signal reconstruction at each level
    """

    prices = data.values if hasattr(data, 'values') else data
    original_signal = prices

    # Perform DWT
    coeffs = pywt.wavedec(original_signal, wavelet, level=level)

    # Create figure for multi-resolution analysis
    fig, axes = plt.subplots(level + 2, 1, figsize=(14, 3 * (level + 2)))

    # Plot original signal
    axes[0].plot(original_signal, 'b-', linewidth=2)
    axes[0].set_title(f'{ticker} - Original Signal', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)

    # Plot approximation at each level
    for i in range(level + 1):
        # Create coefficient copy
        coeffs_copy = [c.copy() for c in coeffs]

        # Set all detail coefficients beyond current level to zero
        for j in range(1, min(i + 1, len(coeffs))):
            coeffs_copy[-j] = np.zeros_like(coeffs[-j])

        # Reconstruct signal
        recon = pywt.waverec(coeffs_copy, wavelet)
        min_len = min(len(original_signal), len(recon))

        # Plot
        axes[i + 1].plot(recon[:min_len], 'g-', linewidth=1.5)

        if i == 0:
            title = f'Approximation at Level {level} (cA{level})'
        else:
            title = f'Reconstruction up to Level {level - i} (cA{level} + cD{level}...cD{level - i + 1})'

        axes[i + 1].set_title(title, fontsize=11)
        axes[i + 1].set_ylabel('Price')
        axes[i + 1].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Point')
    plt.tight_layout()
    # plt.show()
    plt.savefig('wavelet_transform.png', dpi=300)


# Main execution
if __name__ == "__main__":
    import matplotlib
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    matplotlib.use('Agg')  # Use a non-interactive backend

    import pandas as pd
    import os

    # 读取Excel数据
    dfs = {
        f"{file.rstrip('.xlsx')}": pd.read_excel(f'../data/{file}', engine='openpyxl', dtype_backend='numpy_nullable')
        for file in os.listdir('../data')
    }
    print(dfs.keys())
    # for df in dfs:
    #     print(df.head())
    ticker = 'TS.CFE'

    # 定义需要绘制的列
    columns_to_plot = ['开盘价(元)', '最高价(元)', '最低价(元)', '收盘价(元)', '结算价', '成交额(百万)', '成交量(股)', '持仓量']

    prices = [row['开盘价(元)'] for _, row in dfs[ticker].iterrows()][:16]
    print(prices)
    # Z-score标准化
    prices_array = np.array(prices).reshape(-1, 1)
    # scaler = StandardScaler()
    # standardized_prices = scaler.fit_transform(prices_array).flatten()

    # 或者Min-Max标准化
    scaler = MinMaxScaler()
    normalized_prices = scaler.fit_transform(prices_array).flatten()

    # 创建图表和子图 - 修改为8行4列
    fig, axes = plt.subplots(nrows=8, ncols=4, figsize=(20, 30))

    # 2. 使用小波变换分析收盘价
    print("\n正在进行小波变换分析...")

    # result = dwt_analysis_and_visualization(
    #     normalized_prices,
    #     ticker=ticker,
    #     wavelet='haar',  # Try also: 'bior1.3', 'coif3', 'sym5'
    #     level=6  # Decomposition level
    # )
    matrix = dwt_visualization(
        normalized_prices,
        ticker=ticker,
        wavelet='haar',  # Try also: 'bior1.3', 'coif3', 'sym5'
        level=4  # Decomposition level
    )
    # print(matrix)
    print("小波系数矩阵 (保留4位小数):")
    for i, row in enumerate(matrix):
        formatted_row = [f"{val:.4f}" for val in row]
        print(f"Level {i:2d}: {' '.join(formatted_row)}")

    # # 3. Additional: Multi-resolution analysis
    # print("\n" + "=" * 60)
    # print("MULTI-RESOLUTION ANALYSIS")
    # print("=" * 60)
    #
    # multi_resolution_analysis(
    #     normalized_prices,
    #     ticker=ticker,
    #     wavelet=result['wavelet'],
    #     level=result['level']
    # )

    # # 4. Print summary statistics
    # print("\n" + "=" * 60)
    # print("ANALYSIS SUMMARY")
    # print("=" * 60)
    # print(f"Wavelet type: {result['wavelet']}")
    # print(f"Decomposition level: {result['level']}")
    # print(f"Original signal length: {len(result['original'])}")
    # print(f"Reconstruction MSE: {result['mse']:.6f}")
    # print(f"Perfect reconstruction: {result['mse'] < 1e-10}")
    #
    # # 5. Coefficient statistics
    # print("\nCoefficient Statistics:")
    # for i, coeff in enumerate(result['coeffs']):
    #     level_name = f"cA{result['level']}" if i == 0 else f"cD{result['level'] - i + 1}"
    #     print(f"{level_name}: mean={coeff.mean():.4f}, std={coeff.std():.4f}, "
    #           f"min={coeff.min():.4f}, max={coeff.max():.4f}")
