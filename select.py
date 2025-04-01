import os
import random
import shutil

def split_dataset(source_dir, train_dir, val_dir, train_ratio=0.8):
    """
    按照给定比例随机抽取文件，将其分成训练集和验证集。

    :param source_dir: 源文件夹路径，包含所有文件
    :param train_dir: 训练集文件夹路径
    :param val_dir: 验证集文件夹路径
    :param train_ratio: 训练集比例，默认为0.8 (80% 用于训练)
    """
    # 确保目标文件夹存在
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 获取所有文件
    files = os.listdir(source_dir)
    total_files = len(files)

    # 按照比例随机打乱文件并划分
    random.shuffle(files)
    train_count = int(total_files * train_ratio)

    # 划分训练集和验证集
    train_files = files[:train_count]
    val_files = files[train_count:]

    # 复制训练集文件
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

    # 复制验证集文件
    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, file))

    print(f"总文件数: {total_files}")
    print(f"训练集文件数: {len(train_files)}")
    print(f"验证集文件数: {len(val_files)}")

if __name__ == "__main__":
    # 设置源文件夹路径和目标文件夹路径
    source_folder = "./labels"  # 原文件所在文件夹
    train_folder = "./train_files"    # 训练集文件夹
    val_folder = "./val_files"        # 验证集文件夹

    # 自定义比例（80% 训练集，20% 验证集）
    split_dataset(source_folder, train_folder, val_folder, train_ratio=0.8)

