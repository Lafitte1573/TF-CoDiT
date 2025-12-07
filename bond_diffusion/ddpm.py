import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from einops import rearrange, repeat
import json


class CausalLlmForDDPM(nn.Module):
    """自然语言条件注入的LLM-DDPM"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. 文本编码器（处理自然语言条件）
        self.text_encoder = TextConditionEncoder(
            model_name=config['text_model'],
            hidden_dim=config['hidden_dim']
        )

        # 2. 图像tokenization（处理噪声2D表示）
        self.image_tokenizer = ImageTokenizer(
            patch_size=config['patch_size'],
            hidden_dim=config['hidden_dim']
        )

        # 3. 多模态融合LLM
        self.multimodal_llm = MultimodalFusionLLM(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads']
        )

        # 4. 噪声预测头（只处理图像token对应的输出）
        self.noise_predictor = NoisePredictionHead(
            hidden_dim=config['hidden_dim'],
            output_channels=config['n_channels']
        )

        # 5. 扩散过程参数
        self.setup_diffusion_params(config['timesteps'])

    def setup_diffusion_params(self, timesteps):
        """设置扩散过程参数"""
        self.timesteps = timesteps
        self.betas = self._cosine_beta_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def forward(self, noisy_images, timesteps, condition_texts):
        """
        前向传播：预测噪声
        Args:
            noisy_images: (B, C, H, W) 带噪声的多通道2D表示
            timesteps: (B,) 扩散时间步
            condition_texts: list of str, 自然语言条件描述
        Returns:
            predicted_noise: (B, C, H, W) 预测的噪声
        """
        B, C, H, W = noisy_images.shape

        # 1. 编码文本条件
        text_embeddings = self.text_encoder(condition_texts)  # (B, L_text, D)

        # 2. 将图像转换为token序列
        image_tokens = self.image_tokenizer.image_to_tokens(noisy_images)  # (B, L_img, D)

        # 3. 将时间步编码添加到图像token
        time_embeddings = self._get_time_embedding(timesteps)  # (B, D)
        image_tokens = image_tokens + time_embeddings.unsqueeze(1)  # 广播到所有图像token

        # 4. 多模态融合处理
        # 拼接文本和图像token：[CLS] + 文本token + [SEP] + 图像token
        multimodal_output = self.multimodal_llm(
            text_embeddings,
            image_tokens
        )  # (B, L_total, D)

        # 5. 提取图像token对应的输出（用于噪声预测）
        image_token_output = self.multimodal_llm.extract_image_output(
            multimodal_output,
            text_length=text_embeddings.shape[1]
        )  # (B, L_img, D)

        # 6. 预测噪声（只基于图像token输出）
        predicted_noise = self.noise_predictor(image_token_output, (C, H, W))

        return predicted_noise

    def _get_time_embedding(self, timesteps):
        """获取时间步嵌入"""
        # 使用正弦位置编码
        half_dim = self.config['hidden_dim'] // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # 如果hidden_dim是奇数，补零
        if self.config['hidden_dim'] % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))

        return emb

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """余弦噪声调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)


# 这个代码写得不对，不需要 TextEncoder 这个类，直接用 LLM 就行
class TextConditionEncoder(nn.Module):
    """文本条件编码器"""

    def __init__(self, model_name='bert-base-uncased', hidden_dim=768):
        super().__init__()

        # 使用预训练语言模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True
        )

        # 如果tokenizer没有pad_token，添加一个
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.text_model.resize_token_embeddings(len(self.tokenizer))

        # 投影到统一的hidden_dim  ## 这个想法倒是没有错，不能再用 LLM 本来的 hidden space 了，更换一个新的空间
        self.projection = nn.Linear(
            self.text_model.config.hidden_size,
            hidden_dim
        ) if self.text_model.config.hidden_size != hidden_dim else nn.Identity()

    def forward(self, texts):
        """
        编码文本条件
        Args:
            texts: list of str, 自然语言描述
        Returns:
            text_embeddings: (B, L_text, D)
        """
        # 1. Tokenization
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # 移动到模型所在设备
        device = next(self.text_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 2. 获取文本表示
        with torch.no_grad():  # 可以冻结文本模型
            outputs = self.text_model(
                **inputs,
                output_hidden_states=True
            )

        # 使用最后一层隐藏状态
        last_hidden_states = outputs.hidden_states[-1]  # (B, L, D_text)

        # 3. 投影到统一维度
        text_embeddings = self.projection(last_hidden_states)  # (B, L, D)

        return text_embeddings


# 这个倒是不错
class ImageTokenizer(nn.Module):
    """图像tokenization模块"""

    def __init__(self, patch_size=16, hidden_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # 图像分块投影
        self.patch_projection = nn.Conv2d(
            in_channels=3,  # 假设输入是3通道（可以调整）
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def image_to_tokens(self, images):
        """
        将图像转换为token序列
        Args:
            images: (B, C, H, W)
        Returns:
            tokens: (B, L_img, D)
        """
        B, C, H, W = images.shape

        # 如果通道数不是3，先投影到3通道（或直接调整投影层）
        if C != 3:
            # 使用1x1卷积调整通道数
            if not hasattr(self, 'channel_adapter'):
                self.channel_adapter = nn.Conv2d(C, 3, kernel_size=1).to(images.device)
            images = self.channel_adapter(images)

        # 分块投影
        patches = self.patch_projection(images)  # (B, D, H', W')

        # 重塑为序列
        tokens = rearrange(patches, 'b d h w -> b (h w) d')

        # 添加位置编码
        pos_embeddings = self._get_position_embeddings(tokens.shape[1], tokens.device)
        tokens = tokens + pos_embeddings

        return tokens

    def _get_position_embeddings(self, seq_len, device):
        """获取位置编码"""
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=device) *
            -(torch.log(torch.tensor(10000.0)) / self.hidden_dim)
        )

        pos_emb = torch.zeros(seq_len, self.hidden_dim, device=device)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)

        return pos_emb


class MultimodalFusionLLM(nn.Module):
    """多模态融合LLM"""

    def __init__(self, hidden_dim=768, num_layers=6, num_heads=12):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 特殊token嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.sep_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # 多模态Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 模态类型嵌入
        self.text_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.image_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, text_embeddings, image_tokens):
        """
        多模态融合处理
        Args:
            text_embeddings: (B, L_text, D) 文本嵌入
            image_tokens: (B, L_img, D) 图像token
        Returns:
            multimodal_output: (B, L_total, D) 融合后的输出
        """
        B = text_embeddings.shape[0]

        # 1. 添加模态类型嵌入
        text_embeddings = text_embeddings + self.text_type_embedding
        image_tokens = image_tokens + self.image_type_embedding

        # 2. 构建完整序列：[CLS] + 文本 + [SEP] + 图像
        cls_tokens = self.cls_token.expand(B, -1, -1)
        sep_tokens = self.sep_token.expand(B, -1, -1)

        # 计算序列长度
        L_text = text_embeddings.shape[1]
        L_img = image_tokens.shape[1]
        L_total = 1 + L_text + 1 + L_img  # CLS + 文本 + SEP + 图像

        # 构建完整序列
        sequence = torch.cat([
            cls_tokens,  # [CLS]
            text_embeddings,  # 文本
            sep_tokens,  # [SEP]
            image_tokens  # 图像
        ], dim=1)  # (B, L_total, D)

        # 3. 创建注意力掩码
        attention_mask = self._create_attention_mask(L_text, L_img, sequence.device)

        # 4. 通过Transformer编码器
        multimodal_output = self.transformer(
            sequence,
            mask=attention_mask
        )

        return multimodal_output

    def extract_image_output(self, multimodal_output, text_length):
        """
        从多模态输出中提取图像部分
        Args:
            multimodal_output: (B, L_total, D) 完整序列输出
            text_length: 文本token长度
        Returns:
            image_output: (B, L_img, D) 图像token对应的输出
        """
        # 图像部分开始位置：CLS(1) + 文本长度 + SEP(1)
        image_start = 1 + text_length + 1
        image_output = multimodal_output[:, image_start:, :]

        return image_output

    def _create_attention_mask(self, L_text, L_img, device):
        """
        创建注意力掩码
        文本可以关注所有文本，图像可以关注所有文本和图像
        这样设计允许图像token利用文本条件信息
        """
        L_total = 1 + L_text + 1 + L_img

        # 初始化全零掩码
        mask = torch.zeros(L_total, L_total, device=device)

        # 文本-文本注意力（完全连接）
        text_start, text_end = 1, 1 + L_text
        mask[text_start:text_end, text_start:text_end] = 1

        # 文本-图像注意力（文本可以看到图像）
        image_start = text_end + 1
        image_end = L_total
        mask[text_start:text_end, image_start:image_end] = 1

        # 图像-图像注意力（完全连接）
        mask[image_start:image_end, image_start:image_end] = 1

        # 图像-文本注意力（图像可以看到文本）
        mask[image_start:image_end, text_start:text_end] = 1

        # CLS token可以看到所有
        mask[0, :] = 1
        mask[:, 0] = 1

        # SEP token可以看到所有
        mask[text_end, :] = 1
        mask[:, text_end] = 1

        # 转换为bool掩码（0表示被mask）
        mask = mask.bool()

        return mask


class NoisePredictionHead(nn.Module):
    """噪声预测头（只处理图像token）"""

    def __init__(self, hidden_dim=768, output_channels=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels

        # 解码器：将token序列转换回图像
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
        )

        # 图像重构层
        self.image_reconstructor = ImageReconstructor(
            hidden_dim * 4,
            output_channels
        )

    def forward(self, image_token_output, image_shape):
        """
        从图像token输出预测噪声
        Args:
            image_token_output: (B, L_img, D) 图像token输出
            image_shape: (C, H, W) 目标图像形状
        Returns:
            predicted_noise: (B, C, H, W) 预测的噪声图像
        """
        B, L_img, D = image_token_output.shape
        C, H, W = image_shape

        # 1. 解码每个token
        decoded = self.decoder(image_token_output)  # (B, L_img, D*4)

        # 2. 重构为图像
        predicted_noise = self.image_reconstructor(decoded, (C, H, W))

        return predicted_noise


class ImageReconstructor(nn.Module):
    """图像重构模块"""

    def __init__(self, token_dim, output_channels):
        super().__init__()
        self.token_dim = token_dim
        self.output_channels = output_channels

        # 计算patch大小（假设输入图像是正方形）
        self.patch_size = 16  # 与ImageTokenizer中的patch_size对应

        # 将每个token解码为patch
        self.patch_decoder = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, output_channels * self.patch_size * self.patch_size)
        )

    def forward(self, tokens, image_shape):
        """
        将token序列重构为图像
        Args:
            tokens: (B, L_img, D_token) token序列
            image_shape: (C, H, W) 目标图像形状
        Returns:
            image: (B, C, H, W) 重构的图像
        """
        B, L_img, D_token = tokens.shape
        C, H, W = image_shape

        # 1. 解码每个token为patch
        patches_flat = self.patch_decoder(tokens)  # (B, L_img, C*P*P)

        # 2. 重塑为patch网格
        grid_size = int(L_img ** 0.5)  # 假设L_img是完全平方数
        patches = patches_flat.reshape(
            B, grid_size, grid_size, C, self.patch_size, self.patch_size
        )

        # 3. 组合成完整图像
        image = rearrange(
            patches,
            'b h w c ph pw -> b c (h ph) (w pw)'
        )

        # 4. 调整到目标尺寸（如果需要）
        if image.shape[2] != H or image.shape[3] != W:
            image = F.interpolate(
                image,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        return image