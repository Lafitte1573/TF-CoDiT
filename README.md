# TF-DiT: Time Series Synthesis with Diffusion Transformers for Treasury Futures

[![GitHub Stars](https://img.shields.io/github/stars/你的用户名/仓库名?style=social)](https://github.com/你的用户名/仓库名)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗-HuggingFace%20Models-yellow.svg)](https://huggingface.co/你的用户名)

<p align="center">
  <img src="assets/cover.png" alt="UniGen-FinTS Cover" width="800"/>
</p>

**中文** | **[English](docs/README_en.md)**

[//]: # (## ✨ 特性亮点)

[//]: # ()
[//]: # (*   **统一架构**：一个模型同时支持图像/视频理解和生成、视觉问答、多模态对话等任务。)

[//]: # (*   **开源复现**：提供完整的训练代码、预训练权重和数据预处理流程。)

[//]: # (*   **即插即用**：提供简单的API和Gradio Web Demo，无需训练即可快速体验。)

[//]: # (*   **SOTA性能**：在 [MMBench]&#40;https://example.com&#41;、[SEED-Bench]&#40;https://example.com&#41; 等权威多模态基准测试中达到领先水平。)

## 🚀 快速开始

### 环境安装
```bash
git clone https://github.com/你的用户名/仓库名.git
cd 仓库名
pip install -r requirements.txt
```

## 🔥 模型训练

### 训练 VAE 模型
1. **数据准备**：把 PNG、JPG、JPEG 等格式的图片保存在 `data_dir` 目录下
2. **执行训练**：
```bash
torchrun --nproc_per_node=$NUM_GPUS vae/train_vae.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR
```
参数解释：
- `--data_dir`：数据集路径。
- `--output_dir`：模型保存路径，训练好的 VAE 模型包括一个 .safetenses，一个 .bin 文件，以及一个 config.json 文件。

### 使用预训练模型快速推理
```python
from src.modeling import UniGenModel
from src.utils import load_image

model = UniGenModel.from_pretrained("你的用户名/模型名")
image = load_image("assets/example_input.jpg")
response = model.generate("描述这张图片", image_input=image)
print(response)  # 输出：这是一只可爱的猫...
```

### 启动交互式Web Demo
```bash
python src/scripts/web_demo.py --share  # 生成公开链接
```

## 📊 性能评测

我们在多个标准基准测试上的结果如下：

| 数据集 | 任务 | 指标 | 得分 | 排名 |
|--------|------|------|------|------|
| MMBench | 多模态理解 | Accuracy | 85.2 | 1st |
| Seed-Bench | 图像/视频问答 | Accuracy | 78.9 | 2nd |
| COCO Caption | 图像描述 | CIDEr | 135.6 | - |

> 详细评测设置和复现步骤请见 [eval/README.md](eval/README.md)。

## 🖼️ 生成样例画廊

### 1. 视觉推理（图生文）
| 输入图像 | 模型生成描述 |
|----------|--------------|
| ![示例1](assets/example_input.jpg) | “一张宁静的湖边日落风景照，橙色的天空倒映在湖面上，远处有群山和树木的剪影。” |

### 2. 创造性生成（文生图/视频）
**输入文本**：“一个宇航员在火星上骑着自行车，科幻风格。”
**生成结果**：
![生成图像](samples/text_to_vision/astronaut_on_mars.png)

### 3. 多模态对话
**用户**：（上传一张植物图片）“这是什么植物？我应该如何养护它？”
**模型**：“这是绿萝，一种常见的室内观叶植物。养护建议：1. 喜阴凉，避免阳光直射；2. 每周浇水1-2次；3. 可每月施一次稀释的液肥。”

[查看更多样例 →](samples/README.md)

## 🛠️ 训练你自己的模型

如果你想在自己的数据上微调或从头训练：

1. **准备数据**：按 [data/README.md](data/README.md) 格式整理数据。
2. **配置训练**：修改 `configs/train.yaml` 中的超参数。
3. **开始训练**：
```bash
python src/scripts/train.py --config configs/train.yaml
```

## 🤝 如何贡献

我们欢迎任何形式的贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何提交问题、功能请求或拉取请求。

## 📜 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源协议。

## 🙏 致谢

感谢以下项目的启发和代码贡献：
- [LLaVA](https://llava-vl.github.io/)
- [MiniGPT-4](https://minigpt-4.github.io/)
- [OpenAI CLIP](https://openai.com/research/clip)

## 📚 引用

如果我们的工作对你有帮助，请引用：
```bibtex
@article{yourmodel2024,
  title={BonDiffuser: Time Series Synthesis Using Diffusion Models for Treasury Bond Futures},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:你的论文号},
  year={2024}
}
```