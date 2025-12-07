import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM, Qwen2Config, AutoTokenizer
from einops import rearrange, repeat
import math
from typing import Optional, Tuple, List, Union, Dict, Any

# model_name = "/mnt/public/mdl/Qwen/Qwen2.5-7B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# print(tokenizer.vocab_size)
# exit()


class MultiModalQwenConfig(Qwen2Config):
    """扩展Qwen配置以支持多模态"""

    def __init__(
            self,
            image_size: int = 256,
            patch_size: int = 16,
            image_channels: int = 3,
            num_image_tokens: int = 256,
            fusion_hidden_size: int = None,  # 默认与text hidden size相同
            num_fusion_layers: int = 4,
            diffusion_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            vocab_size: int = 151643,
            attention_dropout: float = 0.0,
            hidden_dropout: float = 0.0,
            image_token_id: int = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # 图像配置
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_channels = image_channels
        self.num_image_tokens = num_image_tokens

        # 融合配置
        self.fusion_hidden_size = fusion_hidden_size or self.hidden_size
        self.num_fusion_layers = num_fusion_layers

        # 扩散配置
        self.diffusion_timesteps = diffusion_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # 特殊token
        # self.image_start_token_id = vocab_size  # 假设会扩展
        # self.image_end_token_id = vocab_size + 1
        self.image_token_id = image_token_id

        # 其他配置
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout


class Qwen2MultiModalTokenizer:
    """适配Qwen2的多模态分词器"""

    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # 为Qwen2添加图像相关特殊token
        special_tokens = {
            'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|image|>'],
            'pad_token': '<|pad|>' if self.tokenizer.pad_token is None else self.tokenizer.pad_token
        }

        self.tokenizer.add_special_tokens(special_tokens)

        # 更新特殊token id的获取方式
        self.image_start_token_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.image_token_id = self.tokenizer.convert_tokens_to_ids('<|image|>')

    def prepare_multimodal_input(
            self,
            texts: List[str],
            images: Optional[torch.Tensor] = None,
            max_length: int = 2048,
            patch_size: int = 16
    ) -> Dict[str, torch.Tensor]:
        """准备多模态输入，符合Qwen2的对话格式"""

        formatted_texts = []
        for text in texts:
            # Qwen2的标准对话格式
            formatted = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            formatted_texts.append(formatted)

        # 编码文本
        text_encoding = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # 如果有图像，添加图像标记
        if images is not None:
            batch_size = images.shape[0]
            image_size = images.shape[2]  # 假设输入图像为 (B, C, H, W)
            num_image_tokens = (image_size // patch_size) ** 2

            # 创建图像token占位符
            image_tokens = torch.full(
                (batch_size, num_image_tokens),
                self.image_token_id,
                dtype=torch.long
            )

            # 合并文本和图像token
            combined_ids = torch.cat([text_encoding['input_ids'], image_tokens], dim=1)
            combined_mask = torch.cat([
                text_encoding['attention_mask'],
                torch.ones((batch_size, num_image_tokens), dtype=torch.long)
            ], dim=1)

            # 截断到最大长度
            if combined_ids.shape[1] > max_length:
                combined_ids = combined_ids[:, :max_length]
                combined_mask = combined_mask[:, :max_length]

            return {
                'input_ids': combined_ids,
                'attention_mask': combined_mask,
                # 'text_length': text_encoding['input_ids'].shape[1]
            }

        return text_encoding


class ImagePatchEmbedding(nn.Module):
    """图像patch嵌入层，将图像转换为序列"""

    def __init__(self, config: MultiModalQwenConfig):
        super().__init__()
        self.config = config

        # 计算patch数量
        self.num_patches = (config.image_size // config.patch_size) ** 2

        # Patch投影层
        self.projection = nn.Conv2d(
            config.image_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # LayerNorm
        layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-6)  # Qwen2默认值通常是1e-6
        self.norm = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """将图像转换为patch嵌入"""
        # 输入: (B, C, H, W)
        # 输出: (B, num_patches, hidden_size)

        # 投影到patch嵌入
        x = self.projection(pixel_values)  # (B, hidden_size, H', W')
        x = rearrange(x, 'b d h w -> b (h w) d')  # (B, num_patches, hidden_size)

        # 归一化
        x = self.norm(x)

        return x


class TimeEmbedding(nn.Module):
    """时间步嵌入，用于扩散模型"""

    def __init__(self, hidden_size: int, time_dim: int = 256):
        super().__init__()
        self.time_dim = time_dim
        self.hidden_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """生成时间步嵌入"""
        half_dim = self.time_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.time_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.mlp(emb)


class Qwen2ForMultimodalDDPM(Qwen2ForCausalLM):
    """基于Qwen2的多模态扩散模型"""

    config_class = MultiModalQwenConfig

    def __init__(self, config: MultiModalQwenConfig):
        # 先初始化父类（Qwen2ForCausalLM）
        super().__init__(config)

        # 保存原始配置引用
        self.multimodal_config = config

        # 扩展词汇表以包含图像特殊token
        self.resize_token_embeddings(config.vocab_size + 3)  # 3个新token

        # 图像投影层（将图像转换为与文本嵌入相同维度的序列）
        self.image_patch_embed = ImagePatchEmbedding(config)

        # 时间嵌入（扩散模型需要）
        self.time_embedding = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )

        # # 跨模态融合层（保持与Qwen2的架构一致）
        # self.fusion_layers = nn.ModuleList([
        #     RoPEAwareFusionLayer(config)
        #     for _ in range(config.num_fusion_layers)
        # ])

        # 噪声预测头（适配多通道图像）
        self.noise_pred_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 2, config.image_channels * config.patch_size * config.patch_size)
        )

        # 扩散过程设置
        self.setup_diffusion()

        # # 初始化新添加的权重
        # self.initialization()

    # def initialization(self):
    #     """初始化新增模块的权重"""
    #     # 使用与Qwen2一致的初始化方法
    #     std = self.config.initializer_range
    #
    #     # 初始化投影层
    #     nn.init.normal_(self.image_projection.patch_embed.weight, mean=0.0, std=std)
    #     if self.image_projection.patch_embed.bias is not None:
    #         nn.init.zeros_(self.image_projection.patch_embed.bias)
    #
    #     # 初始化token类型嵌入
    #     nn.init.normal_(self.image_projection.token_type_embedding.weight, mean=0.0, std=std)
    #
    #     # 初始化时间嵌入
    #     for layer in self.time_embedding:
    #         if isinstance(layer, nn.Linear):
    #             nn.init.normal_(layer.weight, mean=0.0, std=std)
    #             if layer.bias is not None:
    #                 nn.init.zeros_(layer.bias)
    #
    #     # 初始化噪声预测头
    #     for layer in self.noise_pred_head:
    #         if isinstance(layer, nn.Linear):
    #             nn.init.normal_(layer.weight, mean=0.0, std=std)
    #             if layer.bias is not None:
    #                 nn.init.zeros_(layer.bias)

    def setup_diffusion(self):
        """设置扩散过程参数"""
        timesteps = self.multimodal_config.diffusion_timesteps

        # 线性beta调度
        self.betas = torch.linspace(
            self.multimodal_config.beta_start,
            self.multimodal_config.beta_end,
            timesteps,
            device=self.device if hasattr(self, 'device') else None
        )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 保存到配置中
        self.register_buffer('diffusion_betas', self.betas)
        self.register_buffer('diffusion_alphas', self.alphas)
        self.register_buffer('diffusion_alphas_cumprod', self.alphas_cumprod)
        self.register_buffer('diffusion_sqrt_alphas_cumprod', self.sqrt_alphas_cumprod)
        self.register_buffer('diffusion_sqrt_one_minus_alphas_cumprod', self.sqrt_one_minus_alphas_cumprod)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            timesteps: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            **kwargs
    ) -> Dict[str, torch.Tensor] | Any:
        """
        修正的前向传播逻辑：
        1. 获取文本和图像占位符的token嵌入
        2. 将图像转换为patch嵌入
        3. 替换input_ids中图像占位符对应的嵌入为图像patch嵌入
        4. 将处理后的嵌入输入LLM
        5. 从LLM输出中提取图像部分的隐藏状态
        6. 使用噪声预测头预测噪声
        """

        if pixel_values is None or timesteps is None:
            # 如果没有图像输入，退回到原始文本生成
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
                **kwargs
            )

        batch_size = pixel_values.shape[0]

        # 1. 获取完整的token嵌入（包括文本和图像占位符）
        token_embeds = self.model.embed_tokens(input_ids)  # (B, L_total, D)

        # 2. 将图像转换为patch嵌入
        image_patch_embeds = self.image_patch_embed(pixel_values)  # (B, num_patches, D)

        # 3. 生成时间嵌入
        # 确保timesteps的形状正确
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = timesteps.repeat(batch_size)

        # 3.1 先计算正弦位置编码
        half_dim = self.time_embedding[0].in_features // 2  # 获取MLP第一层的输入维度的一半
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # 如果维度是奇数，补零
        if half_dim * 2 < self.time_embedding[0].in_features:
            emb = F.pad(emb, (0, 1))

        # 3.2 通过MLP
        time_emb = self.time_embedding(emb)  # (B, D)

        # 4. 添加时间嵌入到图像patch中
        time_emb = time_emb.unsqueeze(1)  # (B, 1, D)
        image_patch_embeds = image_patch_embeds + time_emb

        # 4. 查找input_ids中图像占位符的位置
        # 假设图像占位符token是连续的，并且数量等于num_patches
        image_token_mask = input_ids == self.multimodal_config.image_token_id  # (B, L_total)

        # 确保每个样本有正确的图像占位符数量
        num_patches_per_sample = self.multimodal_config.num_image_tokens

        # 验证图像占位符数量
        for i in range(batch_size):
            image_token_count = image_token_mask[i].sum().item()
            if image_token_count != num_patches_per_sample:
                raise ValueError(
                    f"样本 {i}: 期望 {num_patches_per_sample} 个图像占位符，"
                    f"但找到 {image_token_count} 个"
                )

        # 5. 替换图像占位符嵌入为图像patch嵌入
        # 首先复制token嵌入
        combined_embeds = token_embeds.clone()

        # 对每个样本，找到图像占位符的位置并替换
        for i in range(batch_size):
            # 获取当前样本的图像占位符位置
            image_positions = torch.where(image_token_mask[i])[0]  # (num_patches,)

            # 确保位置数量与patch数量一致
            if len(image_positions) != num_patches_per_sample:
                # 如果数量不一致，只替换前num_patches_per_sample个
                image_positions = image_positions[:num_patches_per_sample]

            # 替换嵌入
            combined_embeds[i, image_positions, :] = image_patch_embeds[i]

        # 6. 通过Qwen2的transformer层
        transformer_outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,  # 使用原始的attention_mask
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        # 获取最后一层隐藏状态
        hidden_states = transformer_outputs.last_hidden_state  # (B, L_total, D)

        # 7. 提取图像部分的隐藏状态（对应图像占位符的位置）
        image_hidden_states = torch.zeros_like(image_patch_embeds)
        for i in range(batch_size):
            image_positions = torch.where(image_token_mask[i])[0]  # (num_patches,)
            image_hidden_states[i] = hidden_states[i, image_positions, :]

        # 8. 预测噪声
        predicted_noise = self.noise_pred_head(image_hidden_states)

        if return_dict:
            return {
                "predicted_noise": predicted_noise,
                "hidden_states": hidden_states,
                "image_hidden_states": image_hidden_states,
                "image_positions": image_token_mask,  # 返回图像位置掩码，用于调试
            }

        return (predicted_noise,)

    def _get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """生成时间步嵌入（与RoPE独立）"""
        half_dim = self.config.hidden_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.config.hidden_size % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb

    def compute_loss(
            self,
            clean_images: torch.Tensor,
            noisy_images: torch.Tensor,
            timesteps: torch.Tensor,
            text_inputs: Dict[str, torch.Tensor],
            noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算扩散损失（适配Qwen2）"""
        if noise is None:
            noise = torch.randn_like(clean_images)

        # 前向扩散
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, noisy_images.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, noisy_images.shape)
        noisy_images = sqrt_alpha * clean_images + sqrt_one_minus_alpha * noise

        # 预测噪声
        outputs = self.forward(
            pixel_values=noisy_images,
            timesteps=timesteps,
            **text_inputs
        )

        predicted_noise = outputs["predicted_noise"]

        # 计算损失
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    def _extract(self, arr, timesteps, shape):
        """从数组中提取对应时间步的值"""
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()

        while len(res.shape) < len(shape):
            res = res[..., None]

        return res.expand(shape)


# 测试代码
if __name__ == "__main__":
    # 加载基础配置
    model_path = "/mnt/public/mdl/Qwen/Qwen2.5-7B-Instruct"
    base_config = Qwen2Config.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # 创建分词器
    tokenizer = Qwen2MultiModalTokenizer(model_path)

    # 创建多模态配置
    config = MultiModalQwenConfig(
        **base_config.to_dict(),  # 继承所有Qwen2配置
        image_size=256,
        patch_size=16,
        image_channels=3,
        num_image_tokens=256,
        num_fusion_layers=4,
        diffusion_timesteps=1000,
        image_token_id=tokenizer.image_token_id,  # 添加图像占位符ID
    )
    # print(config)
    # exit()

    # 创建模型
    model = Qwen2ForMultimodalDDPM(config)
    model.to("cuda")

    print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 测试前向传播
    texts = ["A beautiful sunset over mountains"]
    images = torch.randn(1, 3, 256, 256)
    timesteps = torch.randint(0, 1000, (1,))

    # 准备输入
    inputs = tokenizer.prepare_multimodal_input(texts, images)
    for key, value in inputs.items():
        print(f"输入 {key}: {value.shape}")
    print(torch.sum(inputs['input_ids'] == tokenizer.image_token_id))

    # exit()
    inputs['pixel_values'] = images
    inputs['timesteps'] = timesteps

    # 前向传播
    outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
    print(f"预测噪声形状: {outputs['predicted_noise'].shape}")
    for key, value in outputs.items():
        print(f"输出 {key}: {value.shape}")

