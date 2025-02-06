import os
import random
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(train_dir, label_file, val_dir, val_size=0.2):
    # 创建验证集目录
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # 读取标签文件
    with open(label_file, 'r') as f:
        labels = [line.strip().split() for line in f]

    # 将标签中的图像名和对应的年龄进行拆分
    image_names, ages = zip(*labels)

    # 使用 train_test_split 将数据随机拆分为训练集和验证集
    train_images, val_images = train_test_split(image_names, test_size=val_size, random_state=42)

    # 将验证集的图像复制到验证集目录
    for image_name in val_images:
        image_path = os.path.join(train_dir, image_name)
        if os.path.exists(image_path):
            shutil.copy(image_path, val_dir)

    # 将对应的标签文件也拆分成训练集标签和验证集标签
    train_labels = [f"{image} {age}" for image, age in zip(train_images, ages) if image in train_images]
    val_labels = [f"{image} {age}" for image, age in zip(val_images, ages) if image in val_images]

    # 写入新的训练集和验证集标签文件
    with open('../data/annotations/train.txt', 'w') as f:
        f.write("\n".join(train_labels))

    with open('../data/annotations/val.txt', 'w') as f:
        f.write("\n".join(val_labels))

    print(f"Dataset split completed. {len(val_images)} images moved to validation folder.")
