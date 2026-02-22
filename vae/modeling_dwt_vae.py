import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesVAE(nn.Module):
    def __init__(self, in_channel=4, input_shape=(6, 32), mid_channels=(16, 32), latent_dim=32):
        super(TimeSeriesVAE, self).__init__()

        # self.input_shape = input_shape
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, mid_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels[0], mid_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.AdaptiveAvgPool2d((4, 8)),  # 适应不同输入尺寸
            nn.Flatten(),
        )

        # 潜在空间映射
        hidden_dim = mid_channels[1]*input_shape[0]*input_shape[1]
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # 解码器
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (mid_channels[1], input_shape[0], input_shape[1])),  # 从展平的向量恢复到3D张量
            nn.ConvTranspose2d(mid_channels[1], mid_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels[0], in_channel, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # 输出归一化到[0,1]范围
        )

    def encode(self, x):
        """编码过程"""
        h = self.encoder(x)
        # print("x shape:", x.shape)
        # print("h shape:", h.shape)
        # exit()
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """解码过程"""
        h = self.decoder_input(z)
        h = self.decoder(h)
        return h

    def vae_loss(self, recon_x, x, mu, log_var, kl_weight=1e-4):
        """VAE损失函数"""
        # 重构损失
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return {
            "loss": recon_loss + kl_weight * kl_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        # print("z shape:", z.shape)
        recon_x = self.decode(z)

        loss = self.vae_loss(recon_x, x, mu, log_var)
        return {
            "recon_x": recon_x,
            **loss
        }


if __name__ == '__main__':
    # 创建数据
    batch_size = 32
    input_shape = (4, 6, 32)
    x = torch.randn(batch_size, *input_shape)

    # 创建模型
    model = TimeSeriesVAE(in_channel=4, latent_dim=32)
    model.to("cuda")
    outputs = model(x.to("cuda"))
    for output in outputs:
        print(output.shape)
