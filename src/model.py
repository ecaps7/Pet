import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import torchvision.models as models
from torch.utils.data import DataLoader


# 创建回归模型（使用 ResNet50 预训练模型）
def create_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # 修改最后一层适应回归任务
    return model


# 训练模型
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()  # 使用均方误差作为回归任务的损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, ages in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, ages = images.to(device), ages.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), ages)  # 计算损失
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        # 在验证集上评估模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, ages in val_loader:
                images, ages = images.to(device), ages.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), ages)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")


# 在测试集上评估模型
def evaluate_model(model, test_loader):
    model.eval()
    true_ages = []
    predicted_ages = []

    with torch.no_grad():
        for images, ages in test_loader:
            images, ages = images.to(device), ages.to(device)
            outputs = model(images)
            true_ages.extend(ages.cpu().numpy())
            predicted_ages.extend(outputs.squeeze().cpu().numpy())

    mse = mean_squared_error(true_ages, predicted_ages)
    print(f"Test MSE: {mse:.4f}")
