# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


# 读取训练数据和测试数据
# train_data = pd.read_csv('train.csv', encoding='gbk')
# test_data = pd.read_csv('test.csv', encoding='gbk')
train_data = pd.read_csv('train_CMD.csv', encoding='gbk')
test_data = pd.read_csv('test_CMD.csv', encoding='gbk')
# 分离特征和标签
X_train = train_data.drop(columns=['COST'])
y_train = train_data['COST']
X_test = test_data.drop(columns=['COST'])
y_test = test_data['COST']

# 定义SVM回归模型
svr = SVR()

# 设置参数网格进行网格搜索
param_grid = {
    # 'C': [910000000.0],
    # 'gamma': [3e-9]
    'C': list(np.arange(1e7, 1e8, 1e7)),
    'gamma': list(np.arange(1e-9, 1e-8, 1e-9))
    }
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# 获取最佳参数和MAE
best_params = grid_search.best_params_
best_mae = -grid_search.best_score_

# 输出最佳参数和MAE
print("最佳参数:", best_params)
print("最小MAE:", best_mae)

# 使用最佳参数训练SVM模型
best_svr = SVR(C=best_params['C'], gamma=best_params['gamma'])
best_svr.fit(X_train, y_train)
y_pred_train = cross_val_predict(best_svr, X_train, y_train, cv=5)

# 计算训练数据的性能指标
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)
train_corr = np.corrcoef(y_train, y_pred_train)[0, 1]
pe_train=calculate_custom_pe(y_train, y_pred_train)
MAPE_train=calculate_mape(y_train, y_pred_train)


# 使用模型预测测试数据
y_pred_test = best_svr.predict(X_test)

# 计算测试数据的性能指标
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)
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
# 设置全局字体大小
plt.rcParams.update({'font.size': 20})  # 设置字体大小为14
# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_pred_train, c='red', marker='o', label='Train',s=100)
plt.scatter(y_test, y_pred_test, c='green', marker='s', label='Test',s=100)
min_value = min(min(y_train), min(y_test), min(y_pred_train), min(y_pred_test))
max_value = max(max(y_train), max(y_test), max(y_pred_train), max(y_pred_test))
plt.xlim(min_value , max_value)
plt.ylim(min_value , max_value)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend(loc='upper left')
# plt.title('True Values vs. Predictions')
plt.show()
