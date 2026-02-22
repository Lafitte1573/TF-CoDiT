import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List
import json
import math
import yaml


@dataclass
class ModelConfig:
    """VAE模型配置参数"""
    in_channel: int = 4
    input_shape: Tuple[int, int] = (7, 64)  # (height, width)
    mid_channels: Tuple[int, int] = (16, 32)
    latent_dim: int = 32
    kl_weight: float = 1e-4  # 添加kl_weight到模型配置中

    def __post_init__(self):
        """验证参数有效性"""
        if self.in_channel <= 0:
            raise ValueError("in_channel must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if len(self.mid_channels) != 2:
            raise ValueError("mid_channels must have exactly 2 elements")


@dataclass
class TrainingConfig:
    """训练配置参数"""
    # 数据相关
    data_dir: str = "data/processed"
    duration: int = 64
    batch_size: int = 32
    num_workers: int = 4

    # 训练超参数
    num_epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    # 硬件配置
    use_cuda: bool = True
    fp16: bool = True
    pin_memory: bool = False

    # 输出配置
    output_dir: str = "outputs/vae_for_dwt_d64"
    save_steps: int = 1000
    save_total_limit: int = 3

    # 日志配置
    logging_dir: str = "./logs"
    logging_steps: int = 10
    logging_first_step: bool = True

    def __post_init__(self):
        """验证参数有效性"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


class ConfigManager:
    """配置管理器，支持命令行参数和配置文件"""

    @staticmethod
    def parse_args():
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description="Time Series VAE Training")

        # 模型参数
        parser.add_argument("--in_channel", type=int, default=4, help="输入通道数")
        parser.add_argument("--latent_dim", type=int, default=32, help="潜在空间维度")
        parser.add_argument("--duration", type=int, default=64, help="时间序列长度")

        # 训练参数
        parser.add_argument("--data_dir", type=str, default="data/processed",
                            help="数据目录路径")
        parser.add_argument("--output_dir", type=str,
                            default="/outputs/vae_for_dwt_d64",
                            help="输出目录路径")
        parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
        parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
        parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
        parser.add_argument("--kl_weight", type=float, default=1e-4, help="KL散度权重")

        # 硬件参数
        parser.add_argument("--use_cuda", action="store_true", default=True, help="使用CUDA")
        parser.add_argument("--fp16", action="store_true", default=True, help="使用混合精度训练")
        parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")

        # 其他参数
        parser.add_argument("--config_file", type=str, help="配置文件路径(.yaml/.yml/.json)")
        parser.add_argument("--save_steps", type=int, default=1000, help="保存检查点步数")

        return parser.parse_args()

    @staticmethod
    def load_from_json(config_file: str) -> dict:
        """从JSON文件加载配置"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def load_from_yaml(config_file: str) -> dict:
        """从YAML文件加载配置"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def flatten_config(config_dict: dict) -> dict:
        """展平嵌套的配置字典"""
        flattened = {}

        def _flatten(d, parent_key=''):
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    _flatten(v, new_key)
                else:
                    # 处理布尔值字符串
                    if isinstance(v, str) and v.lower() in ['true', 'false']:
                        v = v.lower() == 'true'
                    flattened[new_key] = v

        _flatten(config_dict)
        return flattened

    @classmethod
    def create_configs(cls, args=None) -> Tuple[ModelConfig, TrainingConfig]:
        """创建模型和训练配置"""
        if args is None:
            args = cls.parse_args()

        # 如果提供了配置文件，优先使用配置文件
        if args.config_file:
            # 根据文件扩展名选择加载方法
            if args.config_file.endswith(('.yaml', '.yml')):
                config_dict = cls.load_from_yaml(args.config_file)
            elif args.config_file.endswith('.json'):
                config_dict = cls.load_from_json(args.config_file)
            else:
                raise ValueError(f"Unsupported config file format: {args.config_file}")

            # 展平配置字典以处理嵌套结构
            flat_config = cls.flatten_config(config_dict)

            # 更新args中的值
            for key, value in flat_config.items():
                # 将带点的键转换为下划线形式以匹配argparse参数
                arg_key = key.replace('.', '_')
                if hasattr(args, arg_key):
                    setattr(args, arg_key, value)
                # 同时也尝试原始键名
                elif hasattr(args, key):
                    setattr(args, key, value)

        # 计算input_shape基于duration
        input_height = int(math.log2(args.duration) + 1)
        input_shape = (input_height, args.duration)

        # 创建模型配置
        model_config = ModelConfig(
            in_channel=args.in_channel,
            input_shape=input_shape,
            latent_dim=args.latent_dim,
            kl_weight=args.kl_weight
        )

        # 创建训练配置
        training_config = TrainingConfig(
            data_dir=args.data_dir,
            duration=args.duration,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=getattr(args, 'weight_decay', 0.01),
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
            use_cuda=args.use_cuda,
            fp16=args.fp16,
            pin_memory=getattr(args, 'pin_memory', False),
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            save_steps=args.save_steps,
            save_total_limit=getattr(args, 'save_total_limit', 3),
            logging_dir=getattr(args, 'logging_dir', "./logs"),
            logging_steps=getattr(args, 'logging_steps', 10),
            logging_first_step=getattr(args, 'logging_first_step', True)
        )

        return model_config, training_config


# 默认配置实例
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()

if __name__ == "__main__":
    # 测试配置管理器
    model_config, training_config = ConfigManager.create_configs()
    print("Model Config:", model_config)
    print("Training Config:", training_config)
