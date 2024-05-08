# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


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
# train_data = pd.read_csv('train_CMD.csv', encoding='gbk')
# test_data = pd.read_csv('test_CMD.csv', encoding='gbk')

# 提取特征和目标列
X_train = train_data.drop(columns=['COST'])
y_train = train_data['COST']
X_test = test_data.drop(columns=['COST'])
y_test = test_data['COST']

# 初始化存储性能指标的列表
mae_values = []
rmse_values = []
r2_values = []
corr_values = []

# 初始化k值范围
k_values = list(np.arange(1, 100, 1))

# k_values = list(range(1, 10))

# 训练和测试不同k值的k-NN模型
for k in k_values:
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(X_train, y_train)
    y_pred_train = knn_regressor.predict(X_train)
    y_pred_test = knn_regressor.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_values.append(mae_train)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_values.append(rmse_train)

    r2_train = r2_score(y_train, y_pred_train)
    r2_values.append(r2_train)

    corr_train = np.corrcoef(y_train, y_pred_train)[0, 1]
    corr_values.append(corr_train)

# 找到MAE最小的k值
best_k = k_values[np.argmin(mae_values)]

# 输出最佳k值
print(f"Best k value based on MAE: {best_k}")

# 绘制MAE随k的变化曲线
plt.figure(figsize=(10, 6))
plt.plot(k_values, mae_values, marker='o')
plt.xlabel('Nearest Neighbors Number(k)', fontsize=20)
plt.ylabel('MAE', fontsize=20)
plt.grid(True)
plt.show()

# 使用最佳k值的模型进行测试
knn_regressor = KNeighborsRegressor(n_neighbors=best_k)
knn_regressor.fit(X_train, y_train)
y_pred_test = knn_regressor.predict(X_test)
y_pred_train = knn_regressor.predict(X_train)
# 计算训练数据的性能指标
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)
train_corr = np.corrcoef(y_train, y_pred_train)[0, 1]
pe_train=calculate_custom_pe(y_train, y_pred_train)
_train=calculate_custom_pe(y_train, y_pred_train)
MAPE_train=calculate_mape(y_train, y_pred_train)

# 计算测试数据性能指标
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2= r2_score(y_test, y_pred_test)
test_corr = np.corrcoef(y_test, y_pred_test)[0, 1]
pe_test=calculate_custom_pe(y_test, y_pred_test)
MAPE_test=calculate_mape(y_test, y_pred_test)

# 输出性能指标
print("训练数据：")
print("均方根误差 (RMSE):", train_rmse)
print("平均绝对误差 (MAE):", train_mae)
print("决定系数 (R2):", train_r2)
print("相关系数:", train_corr)
print("pe:",pe_train)
print("MAPE",MAPE_train)

print("\n测试数据：")
print("均方根误差 (RMSE):", test_rmse)
print("平均绝对误差 (MAE):", test_mae)
print("调整决定系数 (R2):", test_r2)
print("相关系数:", test_corr)
print("pe:",pe_test)
print("MAPE",MAPE_test)

# 绘制训练集与测试集上的散点图
plt.figure(figsize=(10, 8))
plt.scatter(y_train, y_pred_train, c='red', label='Train Data', marker='o')
plt.scatter(y_test, y_pred_test, c='green', label='Test Data', marker='s')
min_value = min(min(y_train), min(y_test), min(y_pred_train), min(y_pred_test))
max_value = max(max(y_train), max(y_test), max(y_pred_train), max(y_pred_test))
plt.xlim(min_value , max_value)
plt.ylim(min_value , max_value)
plt.xlabel('Real Values', fontsize=20)
plt.ylabel('Predicted Values', fontsize=20)
plt.legend(loc='upper left', fontsize=20)
plt.show()
