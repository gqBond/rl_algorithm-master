import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,  output_size):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

def main():
    # 读取CSV文件
    df = pd.read_csv('../virtual dataset_RE_DOI.csv')

    # 提取输入特征和目标变量
    X = df[['Inf_NH4', 'Inf_TP', 'DOI']].values
    y = df[['Eff_NH4', 'Eff_TP']].values

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # 转换为 PyTorch 的张量
    X_train_tensor = torch.Tensor(X_train_scaled)
    y_train_tensor = torch.Tensor(y_train_scaled)
    X_test_tensor = torch.Tensor(X_test_scaled)
    y_test_tensor = torch.Tensor(y_test_scaled)

    # 初始化模型
    input_size = 3  # 输入特征数量，加上DOI
    hidden_size1 = 1024  # 隐藏层神经元数量
    hidden_size2 = 1024  # 隐藏层神经元数量
    output_size = 2  # 输出变量数量（Eff_NH4 和 Eff_TP）

    model = MLPModel(input_size, hidden_size1, hidden_size2, output_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 超参数
    num_epochs = 5000
    batch_size = 32

    # 转换为 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化最低 loss
    best_loss = float('inf')


    # 训练模型
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            # 保存最低 loss 的模型
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), '../best_model.pth')

    # 加载最低 loss 的模型
    best_model = MLPModel(input_size, hidden_size1, hidden_size2, output_size)
    best_model.load_state_dict(torch.load('../best_model.pth'))

    # 测试模型
    best_model.eval()
    with torch.no_grad():
        y_pred = best_model(X_test_tensor)

    # 反向转换预测值
    y_pred_unscaled = scaler_y.inverse_transform(y_pred.numpy())

    # 评估模型性能
    mse = mean_squared_error(y_test, y_pred_unscaled)
    print(f'Mean Squared Error on Test Data: {mse}')

    # 数据可视化
    plt.plot(y_test[:, 0], label='Actual Eff_NH4', marker='o', linestyle='', markersize=5)
    plt.plot(y_test[:, 1], label='Actual Eff_TP', marker='o', linestyle='', markersize=5)
    plt.plot(y_pred_unscaled[:, 0], label='Predicted Eff_NH4', marker='x', linestyle='', markersize=5)
    plt.plot(y_pred_unscaled[:, 1], label='Predicted Eff_TP', marker='x', linestyle='', markersize=5)

    # 添加标签和标题
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title('Regression Performance')

    # 添加图例
    plt.legend()

    plt.show()

    # 使用模型进行预测
    def predict_efficiency(doi_input):
        # 假设 doi_input 是一个包含 'Inf_NH4' 和 'Inf_TP' 的 NumPy 数组
        scaled_input = scaler_X.transform(doi_input)
        input_tensor = torch.Tensor(scaled_input)
        with torch.no_grad():
            eff_output = best_model(input_tensor)
        eff_output_unscaled = scaler_y.inverse_transform(eff_output.numpy())
        return eff_output_unscaled


if __name__ == '__main__':
    main()

