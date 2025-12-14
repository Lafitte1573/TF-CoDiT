import os
import glob
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# 查找所有可能的日志目录
base_dir = '/mnt/public/djx/outputs/Qwen2.5-DDPM-Coco/runs'
log_dirs = glob.glob(os.path.join(base_dir, '*'))

# 过滤出存在的目录
existing_dirs = [d for d in log_dirs if os.path.isdir(d)]
print("可用的日志目录:")
for dir_path in existing_dirs:
    print(f"  {dir_path}")


def safe_extract_tensorboard_logs(log_dir):
    """安全地从TensorBoard日志中提取损失数据"""
    if not os.path.exists(log_dir):
        print(f"错误: 目录 {log_dir} 不存在")
        return None

    try:
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        scalars = {}
        for tag in event_acc.Tags()['scalars']:
            scalars[tag] = event_acc.Scalars(tag)
        return scalars
    except Exception as e:
        print(f"读取日志时出错: {e}")
        return None


def plot_training_loss(log_dir, output_path=None):
    """绘制训练损失曲线"""
    # 安全提取日志数据
    scalars = safe_extract_tensorboard_logs(log_dir)

    if scalars is None:
        return

    # 查找损失相关的标量数据
    loss_tags = [tag for tag in scalars.keys() if 'loss' in tag.lower()]

    if not loss_tags:
        print("未找到损失相关的标量数据")
        return

    # 绘制损失曲线
    plt.figure(figsize=(12, 8))

    for tag in loss_tags:
        data = scalars[tag]
        steps = [d.step for d in data]
        values = [d.value for d in data]

        plt.plot(steps, values, label=tag, marker='o', markersize=2)

    plt.title('Training Loss Curve')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图表
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"损失曲线已保存到: {output_path}")
    else:
        plt.savefig('loss_curve.png')

    plt.close()


plot_training_loss("/mnt/public/djx/outputs/Qwen2.5-DDPM-Coco/runs/Dec13_16-55-12_is-dblzzscigklssph2-devmachine-0")