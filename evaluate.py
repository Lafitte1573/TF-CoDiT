import torch
from modeling_dwt_vae import TimeSeriesVAE
import glob
import numpy as np
import random as rm
import pywt
import matplotlib.pyplot as plt
import matplotlib
import math

matplotlib.use('Agg')


def recover_signal(mats):
    coefficients = []

    for mat in mats:
        channel_coefficients = []
        for i, line in enumerate(mat):
            if i == 0:
                channel_coefficients.append(np.array([line[0]]))
            else:
                channel_coefficients.append(line[::int(len(line) / 2 ** (i - 1))])
        coefficients.append(channel_coefficients)
    return coefficients


def inverse_dwt_transformation(coefficients, wavelet='haar'):
    """
    对小波系数进行逆变换，还原时间信号

    Args:
        coefficients: 小波系数，格式为 [approximation_coeffs, detail_coeffs1, detail_coeffs2, ...]
        wavelet: 使用的小波基
        level: 分解层级
    """
    # 重构信号
    reconstructed_signal = pywt.waverec(coefficients, wavelet)
    return reconstructed_signal


# 计算两个信号的MSE差异
def calculate_mse(signal1, signal2):
    """
    计算两个信号之间的均方误差（MSE）
    """
    # 确保两个信号长度一致
    min_length = min(len(signal1), len(signal2))
    signal1 = signal1[:min_length]
    signal2 = signal2[:min_length]

    mse = np.mean((signal1 - signal2) ** 2)
    return mse


# 也可以计算其他指标
def calculate_metrics(signal1, signal2):
    """
    计算信号质量评估指标
    """
    min_length = min(len(signal1), len(signal2))
    signal1 = signal1[:min_length]
    signal2 = signal2[:min_length]

    mse = np.mean((signal1 - signal2) ** 2)
    rmse = np.sqrt(mse)  # 均方根误差
    mae = np.mean(np.abs(signal1 - signal2))  # 平均绝对误差

    # 信噪比计算
    signal_power = np.mean(signal1 ** 2)
    noise_power = mse
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))  # 添加小值避免除零

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'SNR': snr
    }


# 使用随机潜在向量生成新数据
def generate_new_data(model, num_samples=1, latent_dim=32):
    """
    使用训练好的VAE生成新数据
    """
    with torch.no_grad():
        # 从标准正态分布采样潜在向量
        z = torch.randn(num_samples, latent_dim).to("cuda")

        # 通过解码器生成新数据
        generated_data = model.decode(z)

    return generated_data


def plot_signals(original_signal, noisy_reconstructed_signal, metrics, save_path='dwt_plot.png'):
    """
    绘制原始信号和加噪重建信号的对比图
    """
    plt.figure(figsize=(14, 6))

    # 绘制原始信号
    plt.plot(original_signal, label='Original Signal', color='blue', linewidth=2)

    # 绘制加噪重建信号
    plt.plot(noisy_reconstructed_signal, label='Reconstructed Signal', color='red',
             linewidth=2, linestyle='--', alpha=0.8)

    # 填充两信号之间的差异区域
    plt.fill_between(range(len(original_signal)),
                     original_signal,
                     noisy_reconstructed_signal,
                     alpha=0.2, color='gray', label='Difference')

    title = f"Original vs Reconstructed Signal (t={duration})"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time Point', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 在图中合适位置标注MSE值
    x_loc = 0.05
    plt.text(x_loc, 0.95, f'MSE: {metrics["MSE"]:.6f}',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11, fontweight='bold')
    plt.text(x_loc, 0.89, f'MAE: {metrics["MAE"]:.6f}',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)


def inference(duration=8):
    # 生成新数据
    xs = [mat for fn in glob.glob(f"/mnt/public/djx/data/FinTS/*_{duration}.npy") for mat in np.load(fn)]
    print(len(xs))
    x = xs[0]
    print(x.shape)
    # exit()

    # 加载训练好的模型
    ts_vae = TimeSeriesVAE(in_channel=4, input_shape=(int(math.log2(duration) + 1), duration))

    if duration == 32:
        ts_vae.load_state_dict(
            torch.load(f"/mnt/public/djx/outputs/vae_for_dwt/pytorch_model.bin", map_location="cpu")
        )
    else:
        ts_vae.load_state_dict(
            torch.load(f"/mnt/public/djx/outputs/vae_for_dwt_d{duration}/pytorch_model.bin", map_location="cpu")
        )
    ts_vae.eval()
    ts_vae.to("cuda")

    with torch.no_grad():
        x = torch.tensor(x, dtype=torch.float32).to("cuda")

        mins = x.view(x.shape[0], -1).min(dim=1)[0].view(x.shape[0], 1, 1)
        maxs = x.view(x.shape[0], -1).max(dim=1)[0].view(x.shape[0], 1, 1)
        normal_x = (x - mins) / (maxs - mins + 1e-8)

        outputs = ts_vae(normal_x.unsqueeze(0).to("cuda"))
        recon_x = outputs['recon_x'][0]  # .cpu().numpy()
        print(outputs['loss'], x)
        # print(recon_x)

        recon_x = recon_x * (maxs - mins + 1e-8) + mins
        print(recon_x)

        xs = recover_signal(x.cpu().numpy())
        recon_xs = recover_signal(recon_x.cpu().numpy())

        print(xs[0])
        print(recon_xs[0])

        # 假设我们使用haar小波，level=5
        wavelet = 'haar'

        # 重构原始信号
        original_signal = inverse_dwt_transformation(xs[0], wavelet)
        print(original_signal)

        # 重构重建信号
        reconstructed_signal = inverse_dwt_transformation(recon_xs[0], wavelet)
        print(reconstructed_signal)

        # 计算所有指标
        metrics = calculate_metrics(original_signal, reconstructed_signal)
        print(f"MSE: {metrics['MSE']:.6f}")
        print(f"RMSE: {metrics['RMSE']:.6f}")
        print(f"MAE: {metrics['MAE']:.6f}")
        print(f"SNR: {metrics['SNR']:.2f} dB")

        plot_signals(
            original_signal,
            reconstructed_signal,
            metrics=metrics,
            save_path=f"dwt_plot_d{duration}.png"
        )


if __name__ == '__main__':
    for duration in [8, 16, 32, 64]:
        inference(duration=duration)
