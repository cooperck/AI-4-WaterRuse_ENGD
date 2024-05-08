import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

# 准备训练数据和测试数据的特征和标签
X_train_1 = train_data.drop(columns=['COST'])
X_train = X_train_1.drop(columns=['RECOVERY'])
y_train = train_data['COST']

X_test_1 = test_data.drop(columns=['COST'])
X_test = X_test_1.drop(columns=['RECOVERY'])
y_test = test_data['COST']


# X_train = train_data.drop(columns=['COST'])
# y_train = train_data['COST']
#
# X_test = test_data.drop(columns=['COST'])
# y_test = test_data['COST']



# 使用多元线性回归（MLR）进行回归分析
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算训练集的各项指标
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
correlation_train = np.corrcoef(y_train, y_train_pred)[0, 1]
pe_train=calculate_custom_pe(y_train, y_train_pred)
MAPE_train=calculate_mape(y_train, y_train_pred)

# 计算VIF值（方差膨胀因子）
vif = pd.DataFrame()
vif["Features"] = X_train.columns
vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# 计算测试集的各项指标
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
correlation_test = np.corrcoef(y_test, y_test_pred)[0, 1]
pe_test=calculate_custom_pe(y_test, y_test_pred)
MAPE_test=calculate_mape(y_test, y_test_pred)
# 设置全局字体大小
plt.rcParams.update({'font.size': 20})  # 设置字体大小为14
# 绘制散点图
plt.figure(figsize=(10, 5))
plt.scatter(y_train, y_train_pred, c='red', marker='o', label='Train')
plt.scatter(y_test, y_test_pred, c='green', marker='s', label='Test')
min_value = min(min(y_train), min(y_test), min(y_train_pred), min(y_test_pred))
max_value = max(max(y_train), max(y_test), max(y_train_pred), max(y_test_pred))
plt.xlim(-1, max_value)
plt.ylim(-1, max_value)
# plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], linestyle='--', color='blue')
plt.xlabel('True Values')
plt.ylabel('Predictions')
# plt.title('True Values vs. Predictions')
plt.legend(loc='upper left')
plt.show()

# 打印所有指标和VIF值
print(f"训练数据均方根误差 (RMSE): {np.sqrt(mse_train)}")
print(f"训练数据平均绝对误差 (MAE): {mae_train}")
print(f"训练数据决定系数 (R2): {r2_train}")
print(f"训练数据相关系数: {correlation_train}")
print("rme:",pe_train)
print("MAPE",MAPE_train)

print("VIF值:")
print(vif)
print(f"测试数据调整决定系数 (R2): {r2_test}")
print(f"测试数据相关系数: {correlation_test}")
print(f"测试数据均方根误差 (RMSE): {np.sqrt(mse_test)}")
print(f"测试数据平均绝对误差 (MAE): {mae_test}")
print("rme:",pe_test)
print("MAPE",MAPE_test)
# -*- coding: utf-8 -*-
