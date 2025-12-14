import torch
from datasets import load_dataset

from transformers import Qwen2Config
from bond_diffusion import Qwen2ForMultimodalDDPM, Qwen2MMTokenizer, MultiModalQwenConfig
import os
from os import path

from transformers import Trainer, TrainingArguments
from PIL import Image
from functools import partial

from typing import Optional, Tuple, List, Union, Dict, Any
from diffusion import create_diffusion
from diffusers import AutoencoderKL
import torch.nn as nn
import torch.nn.functional as F

# 使用示例
if __name__ == "__main__":
    # 加载数据集
    dataset = load_dataset(
        "json",
        data_files="/mnt/public/djx/data/CoCo_Caption/train.json"
    )  # 默认就是 split='train'
    print(dataset.column_names)
    train_set = dataset["train"]
    train_set = train_set.map(
        lambda x: {
            "messages": [{
                "prompt": x["label"],
                "image": path.join("/mnt/public/djx/data/CoCo_2017", x["img_path"])
            }],
        },
        remove_columns=train_set.column_names,
        num_proc=os.cpu_count(),
        cache_file_name="/mnt/public/djx/data/CoCo_Caption/caches/train.cache"
    )
    print(len(train_set), train_set.column_names, train_set[0])
    # exit()
    # 1. 创建配置
    # config = MultiModalQwenConfig(
    #     text_model_name_or_path="gpt2",
    #     image_size=256,
    #     patch_size=16,
    #     image_channels=3,
    #     timesteps=1000,
    #     fusion_hidden_size=768,
    #     num_fusion_layers=4,
    #     num_attention_heads=12,
    #     dropout_prob=0.1,
    #     freeze_text_model=True
    # )

    model_id = '/mnt/public/mdl/Qwen/Qwen2.5-1.5B-Instruct'
    base_config = Qwen2Config.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    image_size = 28
    patch_size = 14

    # 2. 创建多模态分词器
    tokenizer = Qwen2MMTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
        image_size=image_size,  # kwargs 参数，一样的传参方法
        patch_size=patch_size,
    )
    print(tokenizer.image_size, tokenizer.patch_size)
    # exit()
    tokenizer.padding_side = "left"
    print(tokenizer.image_token_id)

    # 创建多模态配置
    config = MultiModalQwenConfig(
        **base_config.to_dict(),  # 继承所有Qwen2配置
        image_size=image_size,  # 224（224是原本图像的尺寸，但是训练时用的是潜变量，VAE编码后的潜变量的尺寸是32）
        patch_size=patch_size,
        image_channels=4,
        num_image_tokens=4,  # (28/14)**2
        # num_fusion_layers=4,
        diffusion_timesteps=1000,
        image_token_id=tokenizer.image_token_id,  # 添加图像占位符ID
    )

    # 3. 创建模型
    model = Qwen2ForMultimodalDDPM.from_pretrained(  # 主模型
        model_id,
        config=config,
        trust_remote_code=True,
    )
    model.freeze_parameters()

    vae = AutoencoderKL.from_pretrained(  # VAE
        f"/mnt/public/mdl/Others/sd-vae-ft-mse",
        trust_remote_code=True,
    )

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # DDP 分布式训练不写这个
    # model.to(device)
    # # # model.eval()

    # print(f"Model created on {device}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/sum(p.numel() for p in model.parameters()):.4f}")

    training_args = TrainingArguments(
        output_dir='/mnt/public/djx/outputs/Qwen2.5-DDPM-Coco',
        per_device_train_batch_size=64,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        logging_steps=5,
        save_steps=10_000,
        learning_rate=2.5e-4,  # LoRA训练通常使用更高的学习率
        num_train_epochs=100,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        save_total_limit=4,
        report_to="tensorboard",  # 禁用wandb等报告工具
    )

    def collect_fn(batch, _tokenizer: Qwen2MMTokenizer):
        batch_tokenized = {
            "input_ids": [],
            "attention_mask": [],
            "pixel_values": [],
        }
        for example in batch:
            messages = example["messages"]
            # print(messages)
            text = _tokenizer.apply_chat_template(messages, tokenize=False)
            # print(text)
            # print(messages[0]["image"], Image.open(messages[0]["image"]).convert("RGB"))
            tokenized = _tokenizer(
                text=text,
                image=Image.open(messages[0]["image"]).convert("RGB"),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=384,
            )
            for key, value in tokenized.items():
                batch_tokenized[key].append(value.squeeze(0))
        batch_tokenized = {key: torch.stack(value, dim=0) for key, value in batch_tokenized.items()}
        # for key, value in batch_tokenized.items():
        #     print(key, value.shape)
        return batch_tokenized


    class CustomTrainer(Trainer):
        def compute_loss(
                self,
                model: Qwen2ForMultimodalDDPM,
                inputs: dict[str, Union[torch.Tensor, Any]],
                return_outputs: bool = False,
                num_items_in_batch: Optional[torch.Tensor] = None,
                noise: Optional[torch.Tensor] = None
        ):
            x = inputs.pop("pixel_values")  # 原本的图片根本不使用，只是用来构建一个同尺寸的噪声
            # print(x.shape)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                # print(x.shape)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=model.device)
            # print(t.shape)
            try:
                loss = model.compute_loss(
                    clean_images=x,
                    timesteps=t,
                    text_inputs=inputs
                )
            except Exception as e:
                # print(e)
                loss = model.module.compute_loss(
                    clean_images=x,
                    timesteps=t,
                    text_inputs=inputs
                )

            return loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        # eval_dataset=dataset["validation"],
        data_collator=partial(collect_fn, _tokenizer=tokenizer),
    )

    # 在训练循环开始前，确保VAE与模型在同一设备
    if hasattr(model, 'device'):
        vae = vae.to(model.device)

    # 开始训练
    trainer.train()

    # 只在主进程中执行保存
    if trainer.is_world_process_zero():
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)