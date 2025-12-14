import transformers
from transformers import Trainer, TrainingArguments
from torchvision import transforms
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os

from typing import Union, List

from datasets import load_dataset


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, image_dir: Union[str, os.PathLike, List[Union[str, os.PathLike]]], transform=None):
        self.image_dir = image_dir
        self.transform = transform
        if isinstance(image_dir, str):
            self.image_dir = [image_dir]
        self.images = [
            os.path.join(image_dir, f) for image_dir in self.image_dir for f in os.listdir(image_dir) if
            f.endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.image_dir, self.images[idx])
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image}


#
#
# dataset = load_dataset(
#     "imagefolder",
#     data_files=[
#         "/mnt/public/djx/data/CoCo_2014/train2014",
#         "/mnt/public/djx/data/CoCo_2017/train2017",
#     ],
#     split="train",
#     # cache_dir="/mnt/public/djx/cache",
# )
# print(dataset.column_names, len(dataset), dataset[0])
# exit()


# 数据预处理
transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

# 初始化VAE模型
vae = AutoencoderKL(
    in_channels=3,
    out_channels=3,
    down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
    up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
    block_out_channels=[128, 256, 512, 512],
    layers_per_block=2,
    act_fn="silu",
    latent_channels=4,
    sample_size=256,
)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自定义模型包装类
class VAEWrapper(nn.Module):
    def __init__(self, vae_model, kl_weight=1e-4, use_l1_loss=False):
        super().__init__()
        self.vae = vae_model
        self.kl_weight = kl_weight
        self.use_l1_loss = use_l1_loss

    def forward(self, pixel_values):
        # 编码
        encoded = self.vae.encode(pixel_values)

        # 重参数化采样
        latent = encoded.latent_dist.sample()
        mean = encoded.latent_dist.mean
        logvar = encoded.latent_dist.logvar

        # 解码
        decoded = self.vae.decode(latent).sample

        # 计算重构损失（可选择MSE或L1）
        if self.use_l1_loss:
            recon_loss = nn.functional.l1_loss(
                decoded, pixel_values, reduction='none'
            )
        else:
            recon_loss = nn.functional.mse_loss(
                decoded, pixel_values, reduction='none'
            )

        # 对每个样本平均
        recon_loss = recon_loss.mean(dim=[1, 2, 3]).mean()

        # 计算KL散度（对每个样本求和后平均）
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mean.pow(2) - logvar.exp()
        )

        # 总损失
        loss = recon_loss + self.kl_weight * kl_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "reconstruction": decoded,
            "latent_mean": mean,
            "latent_logvar": logvar
        }


# 包装模型
wrapped_vae = VAEWrapper(vae)  # .to(device)


# # 如果使用多GPU，使用 DistributedDataParallel 而不是 DataParallel
# if torch.cuda.device_count() > 1:
#     from torch.nn.parallel import DistributedDataParallel as DDP
#     wrapped_vae = DDP(wrapped_vae)


# 自定义训练器
class VAETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs["pixel_values"])
        loss = outputs["loss"]

        # 记录额外的损失项用于监控
        if return_outputs:
            outputs["recon_loss"] = outputs.get("recon_loss", torch.tensor(0.0))
            outputs["kl_loss"] = outputs.get("kl_loss", torch.tensor(0.0))

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        # 添加自定义日志记录
        if "recon_loss" in logs:
            logs["recon_loss"] = float(logs["recon_loss"])
        if "kl_loss" in logs:
            logs["kl_loss"] = float(logs["kl_loss"])
        super().log(logs)


# 加载数据集 (你需要指定自己的图像目录)
dataset = ImageDataset(
    image_dir=[
        # "/mnt/public/djx/data/CoCo_2014/train2014",
        "/mnt/public/djx/data/CoCo_2017/train2017",
    ],
    transform=transform
)

# 训练参数配置
training_args = TrainingArguments(
    output_dir="/mnt/public/djx/outputs/vae_sft",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=5,
    save_steps=1000,
    save_total_limit=3,
    logging_first_step=True,
    logging_strategy="steps",
    remove_unused_columns=False,
    dataloader_num_workers=4,
    fp16=torch.cuda.is_available(),  # 如果有GPU则启用混合精度训练
    # 改进分布式训练参数
    dataloader_pin_memory=False,
    ddp_find_unused_parameters=False,
    # # 添加以下参数以提高稳定性
    # sharded_ddp=False,  # 如果不需要分片则关闭
    report_to="tensorboard",
)

# # 在创建 trainer 之前添加
# if torch.cuda.device_count() > 1:
#     wrapped_vae = torch.nn.DataParallel(wrapped_vae)
# # wrapped_vae = wrapped_vae.to(device)

# 初始化训练器
trainer = VAETrainer(
    model=wrapped_vae,
    args=training_args,
    train_dataset=dataset,  # 需要替换为实际的数据集
    tokenizer=None,
)

# 开始训练
trainer.train()

# 保存完整的 VAE 模型（包括配置和权重）
vae.save_pretrained("/mnt/public/djx/outputs/vae_sft/final_vae_model")
