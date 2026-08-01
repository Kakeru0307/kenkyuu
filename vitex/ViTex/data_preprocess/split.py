import os
import random
import shutil

def split_pkl_files(src_dir, val_dir, ratio=0.1, seed=42):
    """
    从src_dir中抽取一定比例的pkl文件，移动到val_dir作为验证集

    :param src_dir: 原始文件夹路径
    :param val_dir: 验证集文件夹路径
    :param ratio: 抽取比例，默认0.1
    :param seed: 随机种子，保证可复现
    """
    # 创建验证集文件夹
    os.makedirs(val_dir, exist_ok=True)

    # 获取所有pkl文件
    files = [f for f in os.listdir(src_dir) if f.endswith('.pkl')]
    total = len(files)
    num_val = max(1, int(total * ratio))  # 至少抽一个

    # 随机抽样
    random.seed(seed)
    val_files = random.sample(files, num_val)

    # 移动文件
    for f in val_files:
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(val_dir, f)
        shutil.move(src_path, dst_path)

    print(f"Total file nums: {total}")
    print(f"Moving {num_val} files to {val_dir}")

# 示例调用
if __name__ == "__main__":
    split_pkl_files("/workspace/midi_pkl", "/workspace/midi_pkl_test", ratio=0.1)
