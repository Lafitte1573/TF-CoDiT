import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np


def generate_gaussian_noise_image(size=(224, 224), mean=0, std=1):
    """
    生成高斯白噪声图片

    Args:
        size: 图片尺寸 (height, width)
        mean: 高斯分布均值
        std: 高斯分布标准差

    Returns:
        PIL Image: 高斯白噪声图片
    """
    # 生成随机高斯噪声 (H, W, 3)
    noise = torch.randn(size[0], size[1], 3) * std + mean

    # 将值限制在 [0, 1] 范围内
    noise = torch.clamp(noise, 0, 1)

    # 转换为 PIL Image
    transform = T.ToPILImage()
    noise_image = transform(noise.permute(2, 0, 1))  # 调整维度顺序 (C, H, W)

    return noise_image


# 使用示例
if __name__ == "__main__":
    # 生成 224x224 的高斯白噪声图片
    noise_img = generate_gaussian_noise_image((224, 224))
    noise_img.save("gaussian_noise.png")
    print("高斯白噪声图片已保存为 gaussian_noise.png")
