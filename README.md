<h1 style="text-align: center">TF-CoDiT: Conditional Time Series Synthesis with Diffusion Transformers for Treasury Futures</h1>

---

<p align="center">
  <a>Anonymous Authors</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-2601.11880-b31b1b.svg?style=flat-square&logo=arxiv" alt="arXiv" />
  <img src="https://img.shields.io/badge/Python-3.10-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square" alt="License" />

[//]: # (  <img src="https://img.shields.io/github/stars/Lafitte1573/TF-DiT?style=flat-square&logo=github" alt="Stars" />)
[//]: # (  <img src="https://img.shields.io/github/stars/Lafitte1573/TF-CoDiT?style=flat-square&logo=github" alt="Stars" />)
</p>

<div align="center">
<img src="plot/cover.png" width="80%">
</div>

---

## Contents

- [Setup](#-setup)
- [Data Preparation](#-data-preparation)
- [Training](#-training)
- [Inference](#-inference)
- [Evaluation](#-evaluation)
- [Citation](#-citation)
- [Contact & Acknowledgements](#-contact--acknowledgements)

---

## Setup

```bash
# Create environment (Python ~3.10)
conda create -n your-project python=3.10
conda activate your-project

# Clone repository
git clone https://github.com/username/repo.git
cd repo

# Install dependencies
pip install -r requirements.txt
```
---

## Data Preparation

- Download the data from [here](https://drive.google.com/drive/folders/1YZx0Y_5X-Y5QXwzq_qy7XwQ5Xwzq_qy7XwQ5Xwzq_qy7XwQ5Xwzq_qy7XwQ5Xwzq_qy7XwQ5Xwzq_qy7XwQ5Xwzq_qy7XwQ5Xwzq_qy7XwQ5Xwzq_qy7)
- Extract the data to `data/`
- Run `python utils/preprocess.py` to preprocess the data
- Run `python utils/prepare_dataset.py` to prepare the dataset

---

## Training

| Device | Command |
|--------|---------|
| **TPU** | `python train.py -c configs/your_config.yaml` (requires gcloud + env setup) |
| **GPU** | `deepspeed train.py -c configs/your_config.yaml` |

Configs live in `configs/`. Adjust `batch_size`, data paths, etc. as needed.

---

## Inference

**1. Convert checkpoint to diffusers pipeline:**

```bash
python utils/save_pipeline.py
  --checkpoint /path/to/checkpoint/ \
  --trainer spmd \
  --type fuse-dit \
  --compression
```

**2. Run inference:**

```bash
python inference.py 
  --checkpoint_path /path/to/pipeline/ \
  --prompt "your prompt" \
  --resolution 512 \
  --num_inference_steps 25 \
  --guidance_scale 6.0 \
  --save_path out.jpg
```

---

[//]: # ()
[//]: # (## Evaluation)

[//]: # ()
[//]: # (| Benchmark | Command |)

[//]: # (|-----------|---------|)

[//]: # (| **GenEval** | `accelerate launch evaluation/sample_geneval.py evaluation/geneval.yaml` |)

[//]: # (| **DPG-Bench** | `accelerate launch evaluation/sample_dpgbench.py evaluation/dpgbench.yaml` |)

[//]: # (| **FID &#40;MJHQ-30K&#41;** | `accelerate launch evaluation/sample_mjhq.py ...` + `python evaluation/fid.py ...` |)

[//]: # ()
[//]: # (See each benchmark’s official repo and scripts in `evaluation/` for full steps.)
[//]: # ()
[//]: # (---)

## Citation

```bibtex
@article{author2025title,
  title  = {Your Paper Title},
  author = {Author One and Author Two},
  year   = {2025},
  journal = {arXiv preprint arXiv:xxxx.xxxxx}
}
```