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
y_train = torch.Tensor(y_train.values)
X_test = torch.Tensor(X_test.values)
y_test = torch.Tensor(y_test.values)
# 存储损失函数
loss_list=[]
# 定义神经网络模型
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, activation):
        super(RegressionModel, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_size, hidden_size))
        if num_hidden_layers > 1:
            for _ in range(num_hidden_layers - 1):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers = nn.ModuleList(self.layers)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

# 训练函数
def train_model(model, X, y, num_epochs=20000, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
            loss_list.append(loss)

# 计算RMSE
def calculate_rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

# 创建不同神经网络结构的模型并训练
best_rmse = float('inf')
best_model = None
results = []
# 'sigmoid', 'tanh', 'relu', 'leaky_relu'
for activation in ['sigmoid', 'tanh', 'relu', 'leaky_relu']:
    for num_hidden_layers in range(5, 10):
        for hidden_size in range(25, 30):
            model = RegressionModel(X_train.shape[1], hidden_size, num_hidden_layers, activation)
            train_model(model, X_train, y_train)
            y_pred_train = model(X_train)
            y_pred_train = model(X_train).squeeze()  # 降维以匹配维度
            print('ytrain',y_train)
            print('ytrainpred',y_pred_train.detach().numpy())
            mae = mean_absolute_error(y_train, y_pred_train.detach().numpy())
            rmse = calculate_rmse(y_train, y_pred_train.detach().numpy())
            r2 = r2_score(y_train, y_pred_train.detach().numpy())
            corr = np.corrcoef(y_train, y_pred_train.detach().numpy())[0, 1]
            pe = calculate_custom_pe(y_train, y_pred_train.detach().numpy())
            MAPE = calculate_mape(y_train, y_pred_train.detach().numpy())

            results.append((activation, num_hidden_layers, hidden_size, mae, rmse, r2, corr,pe,MAPE))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model

# 输出最佳模型的结构和性能
best_activation, best_num_hidden_layers, best_hidden_size, best_mae, best_rmse, best_r2, best_corr,best_pe,best_MAPE = results[0]
for result in results:
    activation, num_hidden_layers, hidden_size, mae, rmse, r2, corr ,rme, MAPE= result
    if rmse < best_rmse:
        best_activation, best_num_hidden_layers, best_hidden_size, best_mae, best_rmse, best_r2, best_corr,best_pe,best_MAPE  = result

print(f'Best Activation Function: {best_activation}')
print(f'Best Number of Hidden Layers: {best_num_hidden_layers}')
print(f'Best Hidden Layer Size: {best_hidden_size}')
print(f'Best MAE: {best_mae}')
print(f'Best RMSE: {best_rmse}')
print(f'Best R^2: {best_r2}')
print(f'Best pe: {best_pe}')
print(f'Best MAPE: {best_MAPE}')

# 测试模型
y_pred_test = best_model(X_test).squeeze() # 降维以匹配维度
test_mae = mean_absolute_error(y_test, y_pred_test.detach().numpy())
test_rmse = calculate_rmse(y_test, y_pred_test.detach().numpy())
test_r2 = r2_score(y_test, y_pred_test.detach().numpy())
test_corr = np.corrcoef(y_test, y_pred_test.detach().numpy())[0, 1]
pe_test=calculate_custom_pe(y_test, y_pred_test.detach().numpy())
MAPE_test=calculate_mape(y_test,y_pred_test.detach().numpy())

# 输出测试数据的性能指标
print(f'Test MAE: {test_mae}')
print(f'Test RMSE: {test_rmse}')
print(f'Test R^2: {test_r2}')
print(f'Test Correlation Coefficient: {test_corr}')
print(f'Test pe: {pe_test}')
print(f'Test MAPE: {MAPE_test}')
# 绘制loss图
plt.figure(figsize=(10, 8))
plt.plot(loss_list, c='blue', label='Loss')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.legend(fontsize=20)
plt.show()
# 绘制散点图
plt.figure(figsize=(10, 8))
plt.scatter(y_train, y_pred_train.detach().numpy(), c='red', label='Train', marker='o')
plt.scatter(y_test, y_pred_test.detach().numpy(), c='green', label='Test', marker='s')
min_value = min(min(y_train), min(y_test), min(y_pred_train), min(y_pred_test))
max_value = max(max(y_train), max(y_test), max(y_pred_train), max(y_pred_test))
plt.xlim(min_value , max_value)
plt.ylim(min_value , max_value)
plt.xlabel('Real Values', fontsize=20)
plt.ylabel('Predicted Values', fontsize=20)
plt.legend(loc='upper left', fontsize=20)
plt.show()
