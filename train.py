import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from einops import rearrange, repeat
import json
from typing import Optional, Tuple, List, Union, Dict
import math
from dataclasses import dataclass


@dataclass
class MultiModalConfig(PretrainedConfig):
    """多模态模型配置"""

    # 文本模型配置
    text_model_name_or_path: str = "gpt2"
    text_hidden_size: int = 768

    # 图像配置
    image_size: int = 256
    patch_size: int = 16
    image_channels: int = 3
    image_token_length: int = 256  # 图像token序列长度

    # 融合层配置
    fusion_hidden_size: int = 768
    num_fusion_layers: int = 2
    num_attention_heads: int = 8
    intermediate_size: int = 3072

    # 扩散模型配置
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # 训练配置
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    use_fp16: bool = False
    freeze_text_model: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保配置属性被正确设置
        for key, value in kwargs.items():
            setattr(self, key, value)


class MultiModalTokenizer:
    """多模态分词器，同时处理文本和图像"""

    def __init__(self, text_model_name="gpt2", image_token_length=256):
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # 添加特殊token
        special_tokens = {
            'pad_token': '[PAD]',
            'image_token': '[IMG]',
            'image_start_token': '[IMG_START]',
            'image_end_token': '[IMG_END]'
        }

        num_added = self.text_tokenizer.add_special_tokens(special_tokens)

        # 图像token id映射
        self.image_token_id = self.text_tokenizer.convert_tokens_to_ids('[IMG]')
        self.image_start_token_id = self.text_tokenizer.convert_tokens_to_ids('[IMG_START]')
        self.image_end_token_id = self.text_tokenizer.convert_tokens_to_ids('[IMG_END]')

        self.image_token_length = image_token_length

    def encode_text(self, texts: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """编码文本"""
        return self.text_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', **kwargs)

    def encode_images(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """编码图像为token序列占位符"""
        batch_size = images.shape[0]

        # 创建图像token占位符 [IMG_START] + [IMG] * n + [IMG_END]
        image_token_ids = torch.full(
            (batch_size, self.image_token_length + 2),
            self.image_token_id,
            dtype=torch.long
        )

        # 设置开始和结束token
        image_token_ids[:, 0] = self.image_start_token_id
        image_token_ids[:, -1] = self.image_end_token_id

        # 创建注意力掩码
        attention_mask = torch.ones((batch_size, self.image_token_length + 2), dtype=torch.long)

        return {
            'image_token_ids': image_token_ids,
            'image_attention_mask': attention_mask
        }

    def encode_multimodal(
            self,
            texts: Union[str, List[str]],
            images: Optional[torch.Tensor] = None,
            max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """编码多模态输入"""

        # 编码文本
        text_encoding = self.encode_text(texts, max_length=max_length)
        input_ids = text_encoding['input_ids']
        attention_mask = text_encoding['attention_mask']

        if images is not None:
            # 编码图像
            image_encoding = self.encode_images(images)
            image_token_ids = image_encoding['image_token_ids']
            image_attention_mask = image_encoding['image_attention_mask']

            # 合并文本和图像token
            combined_input_ids = torch.cat([input_ids, image_token_ids], dim=1)
            combined_attention_mask = torch.cat([attention_mask, image_attention_mask], dim=1)

            # 截断到最大长度
            if combined_input_ids.shape[1] > max_length:
                combined_input_ids = combined_input_ids[:, :max_length]
                combined_attention_mask = combined_attention_mask[:, :max_length]

            return {
                'input_ids': combined_input_ids,
                'attention_mask': combined_attention_mask,
                'text_length': input_ids.shape[1],
                'image_length': image_token_ids.shape[1]
            }

        return text_encoding

    @property
    def vocab_size(self):
        return len(self.text_tokenizer)


class ImageProjection(nn.Module):
    """图像投影模块，将图像转换为隐层特征"""

    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config

        # 计算patch数量
        num_patches = (config.image_size // config.patch_size) ** 2

        # Patch投影层
        self.patch_embed = nn.Conv2d(
            config.image_channels,
            config.fusion_hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # 位置编码
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.fusion_hidden_size)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.fusion_hidden_size))

        # 层归一化
        self.layernorm = nn.LayerNorm(config.fusion_hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """将像素值转换为图像特征"""
        batch_size = pixel_values.shape[0]

        # 投影到patch嵌入
        x = self.patch_embed(pixel_values)  # (B, D, H', W')
        x = rearrange(x, 'b d h w -> b (h w) d')  # (B, N, D)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # 添加位置编码
        x = x + self.position_embedding

        # 归一化和dropout
        x = self.layernorm(x)
        x = self.dropout(x)

        return x


class CrossModalAttention(nn.Module):
    """跨模态注意力层"""

    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.self = nn.MultiheadAttention(
            embed_dim=config.fusion_hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout_prob,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(config.fusion_hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)

        # 自注意力
        hidden_states, _ = self.self(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=attention_mask
        )

        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CrossModalFeedForward(nn.Module):
    """跨模态前馈网络"""

    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.fusion_hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.fusion_hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.fusion_hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)

        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


class CrossModalLayer(nn.Module):
    """跨模态层"""

    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.attention = CrossModalAttention(config)
        self.feed_forward = CrossModalFeedForward(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states


class MultiModalFusionModel(nn.Module):
    """多模态融合模型"""

    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config

        # 跨模态层
        self.layers = nn.ModuleList([
            CrossModalLayer(config) for _ in range(config.num_fusion_layers)
        ])

        # 最终归一化
        self.final_layer_norm = nn.LayerNorm(config.fusion_hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class NoisePredictionHead(nn.Module):
    """噪声预测头"""

    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config

        # MLP解码器
        self.mlp = nn.Sequential(
            nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size * 4),
            nn.LayerNorm(config.fusion_hidden_size * 4, eps=config.layer_norm_eps),
            nn.GELU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.fusion_hidden_size * 4, config.fusion_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout_prob),
        )

        # 最终投影到图像patch
        self.patch_proj = nn.Linear(
            config.fusion_hidden_size * 2,
            config.image_channels * config.patch_size * config.patch_size
        )

        self.patch_size = config.patch_size

    def forward(self, hidden_states: torch.Tensor, image_shape: Tuple[int, int, int]) -> torch.Tensor:
        """从隐藏状态预测噪声"""
        B, L, D = hidden_states.shape
        C, H, W = image_shape

        # 通过MLP解码
        x = self.mlp(hidden_states)  # (B, L, D*2)

        # 投影到patch维度
        patches_flat = self.patch_proj(x)  # (B, L, C*P*P)

        # 计算patch网格大小
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size

        # 重塑为图像
        patches = patches_flat.reshape(
            B, grid_h, grid_w, C, self.patch_size, self.patch_size
        )

        # 组合成完整图像
        predicted_noise = rearrange(
            patches,
            'b h w c ph pw -> b c (h ph) (w pw)'
        )

        # 调整大小
        if predicted_noise.shape[2] != H or predicted_noise.shape[3] != W:
            predicted_noise = F.interpolate(
                predicted_noise,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        return predicted_noise


class TimeEmbedding(nn.Module):
    """时间步嵌入"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """生成时间步嵌入"""
        half_dim = self.hidden_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.hidden_size % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.mlp(emb)


class CausalLlmForDDPM(PreTrainedModel):
    """基于因果语言模型的多模态扩散模型"""

    config_class = MultiModalConfig

    def __init__(self, config: MultiModalConfig):
        super().__init__(config)
        self.config = config

        # 初始化文本模型
        self.text_model = AutoModelForCausalLM.from_pretrained(
            config.text_model_name_or_path,
            torch_dtype=torch.float16 if config.use_fp16 else torch.float32
        )

        # 冻结文本模型（可选）
        if config.freeze_text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False

        # 调整token embeddings以包含特殊token
        self.text_model.resize_token_embeddings(self.text_model.config.vocab_size)

        # 图像投影
        self.image_projection = ImageProjection(config)

        # 时间嵌入
        self.time_embedding = TimeEmbedding(config.fusion_hidden_size)

        # 多模态融合
        self.fusion_model = MultiModalFusionModel(config)

        # 噪声预测头
        self.noise_predictor = NoisePredictionHead(config)

        # 扩散过程参数
        self.setup_diffusion()

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 文本模型已经初始化，只初始化新添加的模块
        for module in [self.image_projection, self.fusion_model, self.noise_predictor]:
            if hasattr(module, '_init_weights'):
                module._init_weights()

    def setup_diffusion(self):
        """设置扩散过程参数"""
        timesteps = self.config.timesteps

        # 线性噪声调度
        self.betas = torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            timesteps,
            device=self.device if hasattr(self, 'device') else None
        )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            timesteps: Optional[torch.Tensor] = None,
            text_length: Optional[int] = None,
            image_length: Optional[int] = None,
            return_dict: bool = True,
            **kwargs
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        前向传播
        Args:
            input_ids: 输入的token ids（包含文本和图像占位符）
            attention_mask: 注意力掩码
            pixel_values: 像素值（带噪声的图像）
            timesteps: 扩散时间步
            text_length: 文本token长度
            image_length: 图像token长度
        Returns:
            预测的噪声
        """

        if pixel_values is None or timesteps is None:
            raise ValueError("pixel_values and timesteps must be provided")

        B, C, H, W = pixel_values.shape

        # 1. 处理文本部分
        with torch.set_grad_enabled(not self.config.freeze_text_model):
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        # 获取文本隐藏状态
        text_hidden_states = text_outputs.hidden_states[-1]

        # 2. 投影图像到隐层空间
        image_hidden_states = self.image_projection(pixel_values)

        # 3. 准备融合输入
        # 如果提供了text_length和image_length，分离文本和图像token
        if text_length is not None and image_length is not None:
            # 获取文本部分（去除图像占位符）
            text_features = text_hidden_states[:, :text_length, :]

            # 创建融合特征
            # 格式: [文本特征, 图像特征]
            fusion_input = torch.cat([text_features, image_hidden_states], dim=1)

            # 创建融合注意力掩码
            fusion_attention_mask = None
            if attention_mask is not None:
                # 为图像特征创建掩码（全部为1）
                image_mask = torch.ones(
                    (B, image_hidden_states.shape[1]),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                fusion_attention_mask = torch.cat(
                    [attention_mask[:, :text_length], image_mask],
                    dim=1
                )
        else:
            # 如果未提供长度信息，使用所有特征
            fusion_input = torch.cat([text_hidden_states, image_hidden_states], dim=1)

            # 扩展注意力掩码以包含图像特征
            if attention_mask is not None:
                image_mask = torch.ones(
                    (B, image_hidden_states.shape[1]),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                fusion_attention_mask = torch.cat([attention_mask, image_mask], dim=1)
            else:
                fusion_attention_mask = None

        # 4. 添加时间嵌入
        time_emb = self.time_embedding(timesteps)  # (B, D)
        time_emb = time_emb.unsqueeze(1).expand(-1, fusion_input.shape[1], -1)  # (B, L, D)
        fusion_input = fusion_input + time_emb

        # 5. 多模态融合
        fused_features = self.fusion_model(fusion_input, fusion_attention_mask)

        # 6. 提取图像相关特征进行噪声预测
        # 只使用与图像对应的特征部分
        if text_length is not None:
            image_features = fused_features[:, text_length:, :]
        else:
            # 假设图像特征在序列末尾
            image_features = fused_features[:, -image_hidden_states.shape[1]:, :]

        # 7. 预测噪声
        predicted_noise = self.noise_predictor(image_features, (C, H, W))

        if not return_dict:
            return (predicted_noise,)

        return {"predicted_noise": predicted_noise}

    def q_sample(
            self,
            x_start: torch.Tensor,
            t: torch.Tensor,
            noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            t_index: int,
            text_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """从p(x_{t-1} | x_t)采样"""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        # 预测噪声
        model_output = self.forward(pixel_values=x, timesteps=t, **text_inputs)
        predicted_noise = model_output["predicted_noise"]

        # 计算均值
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(
            self,
            text_inputs: Dict[str, torch.Tensor],
            image_size: Optional[int] = None,
            batch_size: int = 1,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5
    ) -> torch.Tensor:
        """从文本生成图像"""
        if image_size is None:
            image_size = self.config.image_size

        # 准备输入
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # 创建噪声图像
        shape = (
            batch_size,
            self.config.image_channels,
            image_size,
            image_size
        )
        img = torch.randn(shape, device=self.device)

        # 时间步调度
        timesteps = torch.linspace(
            self.config.timesteps - 1, 0, num_inference_steps, device=self.device
        ).long()

        # 逐步去噪
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            img = self.p_sample(img, t_batch, i, text_inputs)

            if i % 10 == 0 or i == len(timesteps) - 1:
                print(f"Step {i}/{len(timesteps)}")

        # 后处理：将值限制在[-1, 1]之间
        img = torch.clamp(img, -1.0, 1.0)

        return img

    def compute_loss(
            self,
            pixel_values: torch.Tensor,
            text_inputs: Dict[str, torch.Tensor],
            noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算扩散损失"""
        B = pixel_values.size(0)

        # 随机采样时间步
        t = torch.randint(0, self.config.timesteps, (B,), device=pixel_values.device).long()

        # 采样噪声
        if noise is None:
            noise = torch.randn_like(pixel_values)

        # 前向扩散
        noisy_images = self.q_sample(pixel_values, t, noise)

        # 预测噪声
        outputs = self.forward(
            pixel_values=noisy_images,
            timesteps=t,
            **text_inputs
        )
        predicted_noise = outputs["predicted_noise"]

        # 计算损失
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    def _extract(self, arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: Tuple[int, ...]) -> torch.Tensor:
        """从给定的张量中提取对应时间步的值"""
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()

        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]

        return res.expand(broadcast_shape)


# 使用示例
if __name__ == "__main__":
    # 1. 创建配置
    config = MultiModalConfig(
        text_model_name_or_path="gpt2",
        image_size=256,
        patch_size=16,
        image_channels=3,
        timesteps=1000,
        fusion_hidden_size=768,
        num_fusion_layers=4,
        num_attention_heads=12,
        dropout_prob=0.1,
        freeze_text_model=True
    )

    # 2. 创建多模态分词器
    tokenizer = MultiModalTokenizer("gpt2", image_token_length=256)

    # 3. 创建模型
    model = CausalLlmForDDPM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model created on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 4. 示例使用
    texts = ["A beautiful sunset over mountains", "A cat sitting on a chair"]

    # 编码多模态输入
    encoding = tokenizer.encode_multimodal(texts)
    print(f"Input IDs shape: {encoding['input_ids'].shape}")

    # 创建伪图像（用于演示）
    batch_size = len(texts)
    fake_images = torch.randn(batch_size, 3, 256, 256).to(device)
    fake_timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

    # 前向传播
    with torch.no_grad():
        outputs = model.forward(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device),
            pixel_values=fake_images,
            timesteps=fake_timesteps,
            text_length=encoding.get('text_length', 10),
            image_length=encoding.get('image_length', 258)
        )

    print(f"Predicted noise shape: {outputs['predicted_noise'].shape}")

    # 5. 保存和加载示例
    # 保存模型
    model.save_pretrained("./multimodal-diffusion-model")
    tokenizer.text_tokenizer.save_pretrained("./multimodal-diffusion-model")

    # 加载模型
    loaded_model = CausalLlmForDDPM.from_pretrained("./multimodal-diffusion-model")
    print("Model loaded successfully!")