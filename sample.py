import torch
from datasets import load_dataset

from transformers import Qwen2Config
from bond_diffusion import Qwen2ForMultimodalDDPM, Qwen2MMTokenizer, MultiModalQwenConfig
import os
from os import path


if __name__ == "__main__":
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

    # 2. 创建多模态分词器
    tokenizer = Qwen2MMTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    print(tokenizer.image_token_id)

    # 创建多模态配置
    config = MultiModalQwenConfig(
        **base_config.to_dict(),  # 继承所有Qwen2配置
        image_size=224,
        patch_size=16,
        image_channels=3,
        num_image_tokens=196,  # (224/16)**2
        # num_fusion_layers=4,
        diffusion_timesteps=1000,
        image_token_id=tokenizer.image_token_id,  # 添加图像占位符ID
    )
    # config.vocab_size = 152064
    # print(config.vocab_size)
    # # exit()

    # 3. 创建模型
    model = Qwen2ForMultimodalDDPM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model created on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 4. 示例使用
    prompts = ["A beautiful sunset over mountains, I like it.", "A cat sitting on a chair"]
    messages = [[{
        "prompt": prompt,
        "image": "gaussian_noise.png"
    }] for prompt in prompts]

    texts = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in messages]
    print(texts)

    inputs = tokenizer(
        texts,
        images=['gaussian_noise.png'] * len(texts),
        return_tensors="pt",
        padding=True,  # 改为布尔值True
        truncation=True,  # 添加截断选项
    )
    for key, value in inputs.items():
        try:
            print(key, value.shape)
        except Exception as e:
            print(e)
            print(key, value)

    print(tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=False))

    # 前向传播
    with torch.no_grad():
        outputs = model.forward(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            pixel_values=inputs['pixel_values'].to(device),  # 采样的时候需要提供一个噪声，但是训练的时候不需要
            timesteps=torch.randint(0, config.diffusion_timesteps, (inputs['input_ids'].shape[0],), device=device),
        )
        for key, value in outputs.items():
            print(key, value.shape)

    print(f"Predicted noise shape: {outputs['predicted_noise'].shape}")

    # # 5. 保存和加载示例
    # # 保存模型
    # model.save_pretrained("./multimodal-diffusion-model")
    # tokenizer.text_tokenizer.save_pretrained("./multimodal-diffusion-model")
    #
    # # 加载模型
    # loaded_model = CausalLlmForDDPM.from_pretrained("./multimodal-diffusion-model")
    # print("Model loaded successfully!")
