# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

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
train_data = pd.read_csv("train.csv", encoding="gbk")
test_data = pd.read_csv("test.csv", encoding="gbk")

# 提取特征和目标列
X_train = train_data.drop("COST", axis=1)
y_train = train_data["COST"]
X_test = test_data.drop("COST", axis=1)
y_test = test_data["COST"]

# 定义随机森林参数范围
param_grid = {
    # 'n_estimators': list(np.arange(400, 460, 10)),
    # 'max_features': ['auto', 'sqrt', 'log2']

    'n_estimators': [420],
    'max_features': ['auto']
}

# 创建随机森林回归模型
rf = RandomForestRegressor(random_state=42)

# 使用GridSearchCV进行参数搜索
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# 输出最佳参数和MAE
best_params = grid_search.best_params_
best_mae = -grid_search.best_score_
print("Best parameters:", best_params)
print("Best MAE:", best_mae)

# 训练最佳模型
best_rf = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_features=best_params['max_features'], random_state=42)
best_rf.fit(X_train, y_train)
# # 使用plot_tree函数绘制第一棵树的结构
# plt.figure(figsize=(40, 20))
# plot_tree(best_rf.estimators_[0], feature_names=X_train.columns, filled=True, rounded=True)
# plt.show()

# 计算训练集和测试集上的性能指标
def calculate_metrics(model, X, y):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    corr = np.corrcoef(y, y_pred)[0, 1]
    pe =calculate_custom_pe(y, y_pred)
    mape=calculate_mape(y, y_pred)
    return rmse, mae, r2, corr, pe, mape

train_rmse, train_mae, train_r2, train_corr, train_pe, train_mape= calculate_metrics(best_rf, X_train, y_train)
test_rmse, test_mae, test_r2, test_corr, test_pe, test_mape = calculate_metrics(best_rf, X_test, y_test)

print("Train RMSE:", train_rmse)
print("Train MAE:", train_mae)
print("Train R2:", train_r2)
print("Train Correlation:", train_corr)
print("Train pe:", train_pe)
print("Train MAPE:", train_mape)

print("Test RMSE:", test_rmse)
print("Test MAE:", test_mae)
print("Test R2:", test_r2)
print("Test Correlation:", test_corr)
print("Test pe:", test_pe)
print("Test MAPE:", test_mape)

# 设置全局字体大小
plt.rcParams.update({'font.size': 20})  # 设置字体大小为14
# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_train, best_rf.predict(X_train), c='red', marker='o', label='Train')
plt.scatter(y_test, best_rf.predict(X_test), c='green', marker='s', label='Test')
min_value = min(min(y_train), min(y_test), min(best_rf.predict(X_train)), min(best_rf.predict(X_test)))
max_value = max(max(y_train), max(y_test), max(best_rf.predict(X_train)), max(best_rf.predict(X_test)))
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.legend(loc="upper left")
plt.show()

'缩短运行时间SHAP部分依赖图可暂时省略'
# # 使用SHAP进行分析
shap.initjs()
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_train)

# 绘制总体特征图
shap.summary_plot(shap_values, X_train, plot_type="bar")
# 绘制每个特征的Shap值热力图
shap.summary_plot(shap_values, X_train)

'缩短运行时间特征部分依赖图可暂时省略'
# 画出各Shapvalue的各特征部分依赖图
explainer2 = shap.TreeExplainer(best_rf)
shap_values2 = explainer2.shap_values(X_train)
# 绘制每个特征的SHAP依赖图
for feature_idx in range(len(X_train.columns)):
    shap.dependence_plot(feature_idx, shap_values2, X_train, interaction_index=None)
plt.show()
'缩短运行时间直接给出重要性排序'
# 计算特征重要性
feature_importance = np.abs(shap_values).mean(axis=0)
sorted_feature_importance = sorted(zip(X_train.columns, feature_importance), key=lambda x: x[1], reverse=True)
sorted_feature_names = [feature[0] for feature in sorted_feature_importance]
print("Feature Importance Ranking:", sorted_feature_names)

"节省运行时间直接输入SHAP排序结果"
sorted_feature_names=['Qin', 'CIin', 'CIout', 'RECOVERY', 'HARDout', 'SSout', 'CODout', 'SSin', 'NH3Nout', 'HARDin',
                      'LEVEL', 'CODin', 'LASin', 'YEAR', 'LASout', 'NH3Nin', 'CHROMin', 'CHROMout']

r2_train_sorted_list=[]
r2_test_sorted_list=[]

"从一个参数到所有参数重新建模分析，得到最佳参数量后直接运行最佳参数量"
for i in range(8,len(sorted_feature_names)+1):
    # 读取训练数据和测试数据：
    X_train_sorted = X_train[sorted_feature_names[:i]]
    print("\n",i,'X_train_sorted:')
    print( X_train_sorted)
    y_train_sorted = y_train

    X_test_sorted = X_test[sorted_feature_names[:i]]
    y_test_sorted = y_test

    # 定义随机森林参数范围
    param_grid_sorted = {
        # 'n_estimators':  list(np.arange(400, 600, 10)),
        # 'max_features': ['auto', 'sqrt', 'log2']

        'n_estimators': [430],
        'max_features': ['auto']
    }

    # 创建随机森林回归模型
    rf_sorted = RandomForestRegressor(random_state=42)

    # 使用GridSearchCV进行参数搜索
    grid_search_sorted = GridSearchCV(rf_sorted, param_grid_sorted, cv=5, scoring='neg_mean_absolute_error')
    grid_search_sorted.fit(X_train_sorted, y_train_sorted)

    # 输出最佳参数和MAE
    best_params_sorted = grid_search_sorted.best_params_
    best_mae_sorted = -grid_search_sorted.best_score_
    print(i,"Best parameters:", best_params_sorted)
    print(i,"Best MAE:", best_mae_sorted)

    # 训练最佳模型
    best_rf_sorted = RandomForestRegressor(n_estimators=best_params_sorted['n_estimators'], max_features=best_params_sorted['max_features'],
                                    random_state=42)
    best_rf_sorted.fit(X_train_sorted, y_train_sorted)

    train_rmse_sorted, train_mae_sorted, train_r2_sorted, train_corr_sorted,train_pe_sorted,train_mape_sorted = calculate_metrics(best_rf_sorted, X_train_sorted, y_train_sorted)
    test_rmse_sorted, test_mae_sorted, test_r2_sorted, test_corr_sorted,test_pe_sorted,test_mape_sorted = calculate_metrics(best_rf_sorted, X_test_sorted, y_test_sorted)


    print(i, "个参数输入训练数据评估结果:")
    print("均方根误差 (RMSE):", train_rmse_sorted)
    print("平均绝对误差 (MAE):", train_mae_sorted)
    print("决定系数 (R2 Score):", train_r2_sorted)
    print("相关系数:", train_corr_sorted)
    print("pe:", train_pe_sorted)
    print("mape:",train_mape_sorted)

    print("\n", i, "个参数输入测试数据评估结果:")
    print("均方根误差 (RMSE):", test_rmse_sorted)
    print("平均绝对误差 (MAE):", test_mae_sorted)
    print("决定系数 (R2 Score):", test_r2_sorted)
    print("相关系数:", test_r2_sorted)
    print("pe:", test_pe_sorted)
    print("mape:", test_mape_sorted)

    r2_train_sorted_list.append(train_r2_sorted)
    r2_test_sorted_list.append(test_r2_sorted)




    # 绘制最佳参数数量时的真实值与预测值的散点图与残差分布图
    '此段输入需要进一步分析的参数数量'
    if i==8: #数字根据需要修改
        print(i,"个参数输入训练数据评估结果:")
        print("均方根误差 (RMSE):", train_rmse_sorted)
        print("平均绝对误差 (MAE):", train_mae_sorted)
        print("决定系数 (R2 Score):", train_r2_sorted)
        print("相关系数:", train_corr_sorted)

        print("\n",i,"个参数输入测试数据评估结果:")
        print("均方根误差 (RMSE):", test_rmse_sorted)
        print("平均绝对误差 (MAE):", test_mae_sorted)
        print("决定系数 (R2 Score):", test_r2_sorted)
        print("相关系数:", test_r2_sorted)

        # 设置全局字体大小
        plt.rcParams.update({'font.size': 20})  # 设置字体大小为14

        # 绘制最佳参数数量时的真实值与预测值的散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(y_train_sorted, best_rf_sorted.predict(X_train_sorted), c='red', marker='o', label='Train')
        plt.scatter(y_test_sorted, best_rf_sorted.predict(X_test_sorted), c='green', marker='s', label='Test')
        min_value_sorted = min(min(y_train_sorted), min(y_test_sorted), min(best_rf_sorted.predict(X_train_sorted)), min(best_rf_sorted.predict(X_train_sorted)))
        max_value_sorted = max(max(y_train_sorted), max(y_test_sorted), max(best_rf_sorted.predict(X_train_sorted)), max(best_rf_sorted.predict(X_train_sorted)))
        plt.xlim(min_value_sorted, max_value_sorted)
        plt.ylim(min_value_sorted, max_value_sorted)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        # plt.title('True Values vs. Predictions')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        # 绘制残差分布图
        # 合并训练集和测试集数据
        all_data = pd.concat([X_train_sorted , X_test_sorted])
        all_labels = pd.concat([y_train_sorted, y_test_sorted])
        # 计算残差
        all_predictions = best_rf_sorted.predict(all_data)
        residuals = all_labels - all_predictions
        # 绘制残差正态分布图
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residual')
        plt.show()
# 设置全局字体大小
plt.rcParams.update({'font.size': 20})  # 设置字体大小为14
# 绘制参数数量与r2之间的关系折点线图
plt.plot(range(1, len(sorted_feature_names) + 1), r2_train_sorted_list, marker='o',color='r',label='Train')
plt.plot(range(1, len(sorted_feature_names) + 1), r2_test_sorted_list,'--', marker='*',color='g',label='Test')
plt.xlabel('Parameters Count')
plt.ylabel('R2')
plt.title('Parameters Count vs. R2')
plt.legend()
plt.show()