"ANN-2.py相较于ANN.py是从后者的运行经验中得到较优的隐藏层与隐藏节点数量，从而提高计算效率"""
# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import numpy as np


# 计算相对中值误差
def calculate_custom_pe(y_actual, y_pred, constant=0.6745):
    """
    Calculate the custom RMSE with a scaling constant.

    Parameters:
    y_actual (array): Actual values.
    y_pred (array): Predicted values.
    constant (float): Scaling constant.

    Returns:
    float: Calculated custom RMSE.
    """
    # Ensure inputs are numpy arrays for element-wise operations
    y_actual = np.array(y_actual)
    y_pred = np.array(y_pred)
    # Avoid division by zero in weights
    y_actual_nonzero = np.where(y_actual == 0, np.finfo(float).eps, y_actual)
    # Calculate the weighted sum of squares
    weighted_sos = np.sum(((y_actual - y_pred) / y_actual_nonzero)** 2)
    # Calculate the mean by dividing by the number of observations minus 1 (degree of freedom adjustment)
    mean_sos = weighted_sos / (len(y_actual) - 1)
    # Multiply by the constant and return the square root
    return constant * np.sqrt(mean_sos)

#计算MAPE平均绝对百分误差
def calculate_mape(observed, predicted):
    # 计算每个数据点的APE
    ape = [abs((obs - pred) / obs) for obs, pred in zip(observed, predicted)]
    # 计算MAPE
    mape = sum(ape) / len(ape)
    return mape
# 读取训练数据和测试数据
train_data = pd.read_csv('train.csv', encoding='gbk')
test_data = pd.read_csv('test.csv', encoding='gbk')

# 提取特征和目标列
X_train = train_data.drop(columns=['COST'])
y_train = train_data['COST']
X_test = test_data.drop(columns=['COST'])
y_test = test_data['COST']

# 转换数据为PyTorch张量
X_train = torch.Tensor(X_train.values)
y_train = torch.Tensor(y_train.values).view(-1, 1)  # 调整形状以匹配模型输出
X_test = torch.Tensor(X_test.values)
y_test = torch.Tensor(y_test.values).view(-1, 1)

# 存储损失函数
loss_list=[]

# 定义神经网络模型
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, activation):
        super(RegressionModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)])
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)

# 训练函数
def train_model(model, X, y, num_epochs=20000, lr=0.0005):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
            loss_list.append(loss.tolist())

# 计算性能指标
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred.detach().numpy())  # 使用detach().numpy()
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred.detach().numpy())  # 使用detach().numpy()
    r2 = r2_score(y_true, y_pred.detach().numpy())  # 使用detach().numpy()
    corr = np.corrcoef(y_true.view(-1).numpy(), y_pred.view(-1).detach().numpy())[0, 1]
    pe = calculate_custom_pe(y_true.view(-1).numpy(), y_pred.view(-1).detach().numpy())
    MAPE = calculate_mape(y_true.view(-1).numpy(), y_pred.view(-1).detach().numpy())

    return rmse, mae, r2, corr,pe, MAPE

# 创建神经网络模型并训练
model = RegressionModel(X_train.shape[1], 23, 15,'relu')
train_model(model, X_train, y_train)

# 计算训练数据性能指标
y_pred_train = model(X_train)
train_rmse, train_mae, train_r2, train_corr, train_pe, train_MAPE = calculate_metrics(y_train, y_pred_train)

# 计算测试数据性能指标
y_pred_test = model(X_test)
test_rmse, test_mae, test_r2, test_corr, test_pe, test_MAPE = calculate_metrics(y_test, y_pred_test)

# 输出性能指标
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Train MAE: {train_mae:.4f}')
print(f'Train R^2: {train_r2:.4f}')
print(f'Train Correlation Coefficient: {train_corr:.4f}')
print(f'Train rme: {train_pe:.4f}')
print(f'Train MAPE: {train_MAPE:.4f}')

print(f'Test RMSE: {test_rmse:.4f}')
print(f'Test MAE: {test_mae:.4f}')
print(f'Test R^2: {test_r2:.4f}')
print(f'Test Correlation Coefficient: {test_corr:.4f}')
print(f'Test rme: {test_pe:.4f}')
print(f'Test MAPE: {test_MAPE:.4f}')

# 绘制loss图
plt.figure(figsize=(10, 8))
plt.plot(loss_list, c='blue', label='Loss')
plt.xlabel('Epoch(×10)', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.legend(fontsize=20)
plt.show()
# # 绘制散点图
# plt.figure(figsize=(10, 8))
# plt.scatter(y_train, y_pred_train.detach().numpy(), c='red', label='Train', marker='o')
# plt.scatter(y_test, y_pred_test.detach().numpy(), c='green', label='Test', marker='s')
# min_value = min(min(y_train), min(y_test), min(y_pred_train.tolist()), min(y_pred_test.tolist()))
# max_value = max(max(y_train), max(y_test), max(y_pred_train.tolist()), max(y_pred_test.tolist()))
# plt.xlim(min_value , max_value)
# plt.ylim(min_value , max_value)
# plt.xlabel('Real Values', fontsize=20)
# plt.ylabel('Predicted Values', fontsize=20)
# plt.legend(loc='upper left', fontsize=20)
# plt.show()
# 绘制散点图
plt.figure(figsize=(10, 8))
plt.scatter(y_train, y_pred_train.view(-1).detach().numpy(), c='red', label='Train Data', marker='o')
plt.scatter(y_test, y_pred_test.view(-1).detach().numpy(), c='green', label='Test Data', marker='s')
min_value = min(min(y_train), min(y_test), min(y_pred_train.view(-1).detach().numpy()), min(y_pred_test.view(-1).detach().numpy()))
max_value = max(max(y_train), max(y_test), max(y_pred_train.view(-1).detach().numpy()), min(y_pred_test.view(-1).detach().numpy()))
plt.xlim(min_value , max_value)
plt.ylim(min_value , max_value)
plt.xlabel('Real Values', fontsize=20)
plt.ylabel('Predicted Values', fontsize=20)
plt.legend(loc='upper left', fontsize=20)
plt.show()

