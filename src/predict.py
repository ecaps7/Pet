import os
import torch
from torch.utils.data import DataLoader
from dataset import PetAgeDataset, get_transforms
from model import create_model
import numpy as np
from sklearn.metrics import mean_absolute_error

def predict_on_new_test_set(model_path, test_dir, test_label_file):
    # 步骤1：加载模型
    model = create_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 步骤2：准备新的测试数据集和数据加载器
    _, val_transform = get_transforms()
    test_dataset = PetAgeDataset(test_dir, test_label_file, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 步骤3：进行预测
    true_ages = []
    predicted_ages = []
    image_names = []

    with torch.no_grad():
        for images, ages in test_loader:
            images, ages = images.to(device), ages.to(device)
            outputs = model(images)
            true_ages.extend(ages.cpu().numpy())
            predicted_ages.extend(outputs.squeeze().cpu().numpy())
            batch_image_names = [name for name, _ in test_dataset.labels[test_loader.dataset.current_index:test_loader.dataset.current_index + len(ages)]]
            image_names.extend(batch_image_names)
            test_loader.dataset.current_index += len(ages)

    mae = mean_absolute_error(true_ages, predicted_ages)
    print(f"Test MAE: {mae:.4f}")

    base_dir = os.path.dirname(os.path.realpath(__file__))
    pred_result_file = os.path.join(base_dir, '../data/annotations/new_pred_result.txt')
    with open(pred_result_file, 'w') as f:
        for name, age in zip(image_names, predicted_ages):
            f.write(f"{name} {int(age)}\n")
    print(f"Prediction results saved to {pred_result_file}")

if __name__ == '__main__':
    model_path = 'pet_age_model.pth'
    test_dir = '../data/testset'  # 新的测试集目录
    test_label_file = '../data/annotations/test.txt'  # 新的测试集标签文件
    predict_on_new_test_set(model_path, test_dir, test_label_file)