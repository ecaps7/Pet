import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 数据清洗函数
def clean_noisy_labels(labels, std_threshold=3):
    ages = np.array([age for _, age in labels])
    mean_age = np.mean(ages)
    std_age = np.std(ages)
    lower_bound = mean_age - std_threshold * std_age
    upper_bound = mean_age + std_threshold * std_age

    clean_labels = []
    for image_name, age in labels:
        if lower_bound <= age <= upper_bound:
            clean_labels.append((image_name, age))

    return clean_labels


# 自定义数据集类
class PetAgeDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None, clean_labels=True):
        self.data_dir = data_dir
        self.transform = transform
        self.labels = []
        self.current_index = 0

        # 读取标签文件
        with open(label_file, 'r') as f:
            for line in f:
                image_name, age = line.strip().split()
                self.labels.append((image_name, int(age)))

        # 清洗噪声标签
        if clean_labels:
            self.labels = clean_noisy_labels(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name, age = self.labels[idx]
        image_path = os.path.join(self.data_dir, image_name)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_name}: {e}")
            return None
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(age, dtype=torch.float32)


# 数据预处理
def get_transforms():
    train_transform = A.Compose([
        A.Resize(height=224, width=224),  # 缩放图片到 224x224
        A.HorizontalFlip(p=0.5),  # 随机水平翻转
        A.VerticalFlip(p=0.2),  # 随机垂直翻转
        A.GaussianBlur(blur_limit=(3, 9), p=0.5),  # 调整模糊度，增大模糊范围
        A.OneOf([
            A.MultiplicativeNoise(multiplier=[0.8, 1.2], p=0.5),  # 增大乘数范围
            A.GaussNoise(mean=0, std=(10.0, 60.0), p=0.5)  # 增大噪声标准差范围
        ], p=0.5),  # 处理噪声
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # 增大亮度和对比度调节范围
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # 颜色抖动
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),  # 弹性变换
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),  # 增大位置调节范围
        A.RandomCrop(height=200, width=200, p=0.3),  # 随机裁剪
        A.Resize(height=224, width=224),  # 裁剪后重新缩放
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ToTensorV2()  # 转换为张量
    ])

    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform


# 创建数据加载器
def get_data_loaders(train_dir, val_dir, test_dir, train_label_file, val_label_file, test_label_file, batch_size):
    train_transform, val_transform = get_transforms()

    train_dataset = PetAgeDataset(train_dir, train_label_file, transform=train_transform)
    val_dataset = PetAgeDataset(val_dir, val_label_file, transform=val_transform)
    test_dataset = PetAgeDataset(test_dir, test_label_file, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
