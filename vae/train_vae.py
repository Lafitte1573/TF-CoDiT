import glob
import math
import os
from typing import Union, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import numpy as np

from .modeling_dwt_vae import TimeSeriesVAE
from .config import ConfigManager, ModelConfig, TrainingConfig


def standardize_channels(image: torch.Tensor, target_mean: float = 0.5,
                        target_std: float = 0.5) -> torch.Tensor:
    """
    在通道维度上对图像做标准化，使每个通道的均值和方差都为指定值

    Args:
        image: torch.Tensor, shape (C, H, W)
        target_mean: 目标均值
        target_std: 目标标准差

    Returns:
        标准化后的图像张量
    """
    standardized_image = torch.zeros_like(image)

    for c in range(image.shape[0]):  # 遍历每个通道
        # 计算当前通道的均值和标准差
        channel_mean = image[c].mean()
        channel_std = image[c].std()

        # 标准化到0均值，1标准差
        normalized = (image[c] - channel_mean) / (channel_std + 1e-8)

        # 调整到目标均值和标准差
        standardized_image[c] = normalized * target_std + target_mean

    return standardized_image


class TimeSeriesDWTDataset(Dataset):
    """
    时间序列DWT数据集类
    """

    def __init__(self, image_dir: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
                 duration: int = 32):
        """
        初始化数据集

        Args:
            image_dir: 图像文件目录路径
            duration: 时间序列长度
        """
        self.image_dir = image_dir
        self.duration = duration

        # 读取 numpy 文件
        pattern = f"{image_dir}/*_{duration}.npy"
        self.file_paths = glob.glob(pattern)

        if not self.file_paths:
            raise ValueError(f"No files found matching pattern: {pattern}")

        # 预加载所有数据到内存
        self.images = []
        for file_path in self.file_paths:
            try:
                data = np.load(file_path)
                self.images.extend(data)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

        if not self.images:
            raise ValueError("No valid images loaded from dataset")

        print(f"Loaded {len(self.images)} samples from {len(self.file_paths)} files")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            包含像素值的字典
        """
        # 获取图像数据
        image = torch.tensor(self.images[idx], dtype=torch.float32)

        # 归一化到[0,1]范围（按通道最小-最大归一化）
        mins = image.view(image.shape[0], -1).min(dim=1)[0].view(image.shape[0], 1, 1)
        maxs = image.view(image.shape[0], -1).max(dim=1)[0].view(image.shape[0], 1, 1)
        standardized_image = (image - mins) / (maxs - mins + 1e-8)

        return {"pixel_values": standardized_image}


class VAETrainer(Trainer):
    """
    自定义VAE训练器
    """

    def compute_loss(self, model: TimeSeriesVAE, inputs: Dict[str, torch.Tensor],
                     return_outputs: bool = False, num_items_in_batch=None):
        """
        计算损失函数

        Args:
            model: VAE模型
            inputs: 输入数据字典
            return_outputs: 是否返回输出
            num_items_in_batch: 批次中项目数

        Returns:
            损失值或(损失值, 输出)元组
        """
        outputs = model(inputs["pixel_values"])
        loss = outputs["loss"]

        # 记录额外的损失项用于监控
        if return_outputs:
            outputs["recon_loss"] = outputs.get("recon_loss", torch.tensor(0.0))
            outputs["kl_loss"] = outputs.get("kl_loss", torch.tensor(0.0))

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict, start_time=None):
        """
        自定义日志记录

        Args:
            logs: 日志字典
            start_time: 开始时间
        """
        # 转换张量为浮点数
        tensor_keys = ["recon_loss", "kl_loss"]
        for key in tensor_keys:
            if key in logs and isinstance(logs[key], torch.Tensor):
                logs[key] = float(logs[key])

        super().log(logs)


def create_training_arguments(training_config: TrainingConfig) -> TrainingArguments:
    """
    创建训练参数

    Args:
        training_config: 训练配置对象

    Returns:
        TrainingArguments实例
    """
    return TrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        logging_dir=training_config.logging_dir,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        logging_first_step=training_config.logging_first_step,
        logging_strategy="steps",
        remove_unused_columns=False,
        dataloader_num_workers=training_config.num_workers,
        fp16=training_config.fp16 and training_config.use_cuda and torch.cuda.is_available(),
        dataloader_pin_memory=training_config.pin_memory,
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
    )


def main():
    """主训练函数"""
    # 解析配置
    model_config, training_config = ConfigManager.create_configs()

    print("=== 模型配置 ===")
    print(f"输入通道数: {model_config.in_channel}")
    print(f"输入形状: {model_config.input_shape}")
    print(f"潜在维度: {model_config.latent_dim}")

    print("\n=== 训练配置 ===")
    print(f"数据目录: {training_config.data_dir}")
    print(f"批次大小: {training_config.batch_size}")
    print(f"训练轮数: {training_config.num_epochs}")
    print(f"学习率: {training_config.learning_rate}")

    # 加载数据集
    try:
        dataset = TimeSeriesDWTDataset(
            image_dir=training_config.data_dir,
            duration=training_config.duration
        )
        print(f"\n数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"错误: 无法加载数据集 - {e}")
        return

    # 创建模型
    model = TimeSeriesVAE(model_config)
    device = "cuda" if training_config.use_cuda and torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"使用设备: {device}")

    # 创建训练参数
    training_args = create_training_arguments(training_config)

    # 初始化训练器
    trainer = VAETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=None,
    )

    # 开始训练
    print("\n开始训练...")
    trainer.train()

    # 保存模型
    model_save_path = os.path.join(training_config.output_dir, "pytorch_model.bin")
    os.makedirs(training_config.output_dir, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\n模型已保存到: {model_save_path}")
    print("训练完成!")


if __name__ == "__main__":
    main()
