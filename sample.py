import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import os
import argparse
import yaml
from pathlib import Path

from diffusion.models import build_model
from vae.modeling_dwt_vae import TimeSeriesVAE
from vae.config import ConfigManager


class DWTSignalSampler:
    def __init__(
        self,
        fusedit_config_path: str,
        fusedit_checkpoint_path: str,
        vae_checkpoint_path: str,
        tokenizer_name: str = "google/gemma-2b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化DWT信号采样器

        Args:
            fusedit_config_path: FuseDiT配置文件路径
            fusedit_checkpoint_path: FuseDiT模型检查点路径
            vae_checkpoint_path: TimeSeriesVAE模型检查点路径
            tokenizer_name: 分词器名称
            device: 设备类型
        """
        self.device = device
        self.tokenizer_name = tokenizer_name

        # 加载FuseDiT模型
        print("Loading FuseDiT model...")
        self.fusedit = self._load_fusedit(fusedit_config_path, fusedit_checkpoint_path)

        # 加载TimeSeriesVAE模型
        print("Loading TimeSeriesVAE model...")
        self.vae = self._load_vae(vae_checkpoint_path)

        # 初始化调度器
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.fusedit.config.noise_scheduler
        )

        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        print(f"Models loaded successfully on {device}")

    def _load_fusedit(self, config_path: str, checkpoint_path: str):
        """加载FuseDiT模型"""
        # 加载配置文件
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 构建模型
        model = build_model(config)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        return model.to(self.device)

    def _load_vae(self, checkpoint_path: str):
        """加载TimeSeriesVAE模型"""
        # 创建VAE配置
        model_config, _ = ConfigManager.create_configs()

        # 构建VAE模型
        vae = TimeSeriesVAE(
            in_channel=model_config.in_channel,
            input_shape=model_config.input_shape,
            mid_channels=model_config.mid_channels,
            latent_dim=model_config.latent_dim
        )

        # 加载检查点
        vae.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        vae.eval()
        return vae.to(self.device)

    def sample(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        generator: torch.Generator = None,
        output_shape: tuple = (7, 64),
        max_length: int = 128
    ) -> torch.Tensor:
        """
        采样生成DWT时序信号矩阵

        Args:
            prompt: 文本提示词
            num_inference_steps: 推理步数
            guidance_scale: 引导尺度
            generator: 随机数生成器
            output_shape: 输出DWT矩阵形状
            max_length: 最大序列长度

        Returns:
            生成的DWT矩阵 (batch_size, channels, height, width)
        """
        # 文本编码
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        batch_size = input_ids.shape[0]

        # 设置调度器timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # 初始化latent噪声
        latent_shape = (batch_size, 4, output_shape[0], output_shape[1])
        latents = torch.randn(
            latent_shape,
            generator=generator,
            device=self.device,
            dtype=torch.float32
        )

        # 扩散过程
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            # classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            timestep = torch.cat([t.reshape(1)] * 2) if guidance_scale > 1.0 else t.reshape(1)

            # 文本条件和无条件输入
            if guidance_scale > 1.0:
                input_ids_cond = torch.cat([input_ids, torch.zeros_like(input_ids)], dim=0)
                attention_mask_cond = torch.cat([attention_mask, torch.zeros_like(attention_mask)], dim=0)
            else:
                input_ids_cond = input_ids
                attention_mask_cond = attention_mask

            # FuseDiT前向传播
            noise_pred = self.fusedit(
                hidden_states=latent_model_input,
                timestep=timestep,
                input_ids=input_ids_cond,
                attention_mask=attention_mask_cond
            )[0]

            # classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 调度器步骤
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 使用VAE解码器将latent转换为DWT矩阵
        with torch.no_grad():
            dwt_matrices = self.vae.decode(latents)

        return dwt_matrices

    def sample_batch(
        self,
        prompts: list,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        generator: torch.Generator = None,
        output_shape: tuple = (7, 64),
        max_length: int = 128
    ) -> torch.Tensor:
        """
        批量采样生成DWT矩阵

        Args:
            prompts: 文本提示词列表
            num_inference_steps: 推理步数
            guidance_scale: 引导尺度
            generator: 随机数生成器
            output_shape: 输出DWT矩阵形状
            max_length: 最大序列长度

        Returns:
            生成的DWT矩阵批量
        """
        # 批量文本编码
        text_inputs = self.tokenizer(
            prompts,
            padding=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        batch_size = len(prompts)

        # 设置调度器
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # 初始化latents
        latent_shape = (batch_size, 4, output_shape[0], output_shape[1])
        latents = torch.randn(
            latent_shape,
            generator=generator,
            device=self.device,
            dtype=torch.float32
        )

        # 扩散过程
        for i, t in enumerate(tqdm(timesteps, desc="Batch Sampling")):
            # classifier-free guidance
            if guidance_scale > 1.0:
                latent_model_input = torch.cat([latents] * 2)
                timestep = torch.cat([t.reshape(1)] * 2)
                input_ids_cond = torch.cat([input_ids, torch.zeros_like(input_ids)], dim=0)
                attention_mask_cond = torch.cat([attention_mask, torch.zeros_like(attention_mask)], dim=0)
            else:
                latent_model_input = latents
                timestep = t.reshape(1)
                input_ids_cond = input_ids
                attention_mask_cond = attention_mask

            # FuseDiT推理
            noise_pred = self.fusedit(
                hidden_states=latent_model_input,
                timestep=timestep,
                input_ids=input_ids_cond,
                attention_mask=attention_mask_cond
            )[0]

            # guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 调度器更新
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # VAE解码
        with torch.no_grad():
            dwt_matrices = self.vae.decode(latents)

        return dwt_matrices


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DWT Signal Sampling with FuseDiT and VAE")

    # 模型路径参数
    parser.add_argument(
        "--fusedit_config",
        type=str,
        required=True,
        help="FuseDiT配置文件路径"
    )
    parser.add_argument(
        "--fusedit_checkpoint",
        type=str,
        required=True,
        help="FuseDiT模型检查点路径"
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        required=True,
        help="TimeSeriesVAE模型检查点路径"
    )

    # 文本提示词参数
    parser.add_argument(
        "--prompt",
        type=str,
        default="generate a financial time series with high volatility",
        help="文本提示词"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="包含多个提示词的文件路径（每行一个提示词）"
    )

    # 采样参数
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="推理步数"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="引导尺度"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    # 模型参数
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="google/gemma-2b",
        help="分词器名称"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="最大序列长度"
    )
    parser.add_argument(
        "--output_shape",
        type=int,
        nargs=2,
        default=[7, 64],
        help="输出DWT矩阵形状 (height width)"
    )

    # 硬件参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="设备类型"
    )

    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./samples",
        help="输出目录"
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="npy",
        choices=["npy", "pt"],
        help="保存格式"
    )

    return parser.parse_args()


def load_prompts_from_file(filepath: str) -> list:
    """从文件加载提示词列表"""
    prompts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def save_samples(samples: torch.Tensor, output_dir: str, format: str = "npy"):
    """保存生成的样本"""
    os.makedirs(output_dir, exist_ok=True)

    if format == "npy":
        np.save(os.path.join(output_dir, "generated_samples.npy"), samples.cpu().numpy())
    elif format == "pt":
        torch.save(samples.cpu(), os.path.join(output_dir, "generated_samples.pt"))

    print(f"Samples saved to {output_dir}")


def main():
    # 解析参数
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    # 创建采样器
    sampler = DWTSignalSampler(
        fusedit_config_path=args.fusedit_config,
        fusedit_checkpoint_path=args.fusedit_checkpoint,
        vae_checkpoint_path=args.vae_checkpoint,
        tokenizer_name=args.tokenizer_name,
        device=args.device
    )

    # 获取提示词
    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = [args.prompt]
        print(f"Using single prompt: {args.prompt}")

    # 采样
    if len(prompts) == 1:
        # 单个提示词采样
        dwt_matrix = sampler.sample(
            prompt=prompts[0],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_shape=tuple(args.output_shape),
            max_length=args.max_length
        )
        samples = dwt_matrix.unsqueeze(0)  # 添加batch维度
        print(f"Generated DWT matrix shape: {dwt_matrix.shape}")
    else:
        # 批量采样
        dwt_batch = sampler.sample_batch(
            prompts=prompts,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_shape=tuple(args.output_shape),
            max_length=args.max_length
        )
        samples = dwt_batch
        print(f"Batch generated DWT matrices shape: {dwt_batch.shape}")

    # 保存结果
    save_samples(samples, args.output_dir, args.save_format)


if __name__ == "__main__":
    main()
