import os
import torch
from torch.utils.data import DataLoader
from dataset import PetAgeDataset, get_transforms
from model import create_model, train_model, evaluate_model
from split_dataset import split_dataset


def main():
    # 数据路径和标签文件路径
    base_dir = os.path.dirname(os.path.realpath(__file__))
    train_dir = os.path.join(base_dir, '../data/trainset')
    val_dir = os.path.join(base_dir, '../data/valset')
    test_dir = os.path.join(base_dir, '../data/testset')
    label_file = os.path.join(base_dir, '../data/annotations/label.txt')
    train_label_file = os.path.join(base_dir, '../data/annotations/train.txt')
    val_label_file = os.path.join(base_dir, '../data/annotations/val.txt')
    test_label_file = os.path.join(base_dir, '../data/annotations/test.txt')

    if not os.path.exists(val_dir):
        print("No validation set found, splitting dataset...")
        split_dataset(train_dir, label_file, val_dir, val_size=0.2)
    else:
        print("Validation set already exists, skipping split.")

    # 获取数据加载器
    train_transform, val_transform = get_transforms()

    train_dataset = PetAgeDataset(train_dir, train_label_file, transform=train_transform)
    val_dataset = PetAgeDataset(val_dir, val_label_file, transform=val_transform)
    test_dataset = PetAgeDataset(test_dir, test_label_file, transform=val_transform, clean_labels=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 创建模型
    model = create_model()

    # 训练模型
    print("Training model...")
    train_model(model, train_loader, val_loader, num_epochs=2, learning_rate=0.001)

    # 评估模型
    print("Evaluating model on test set...")
    evaluate_model(model, test_loader)

    # 保存模型
    torch.save(model.state_dict(), 'pet_age_model.pth')
    print("Model saved to pet_age_model.pth")

if __name__ == '__main__':
    main()
