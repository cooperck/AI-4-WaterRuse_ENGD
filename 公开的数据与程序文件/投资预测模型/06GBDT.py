# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
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
train_data = pd.read_csv('train.csv', encoding='gbk')
test_data = pd.read_csv('test.csv', encoding='gbk')

# 分离特征和标签
X_train = train_data.drop('COST', axis=1)
y_train = train_data['COST']
X_test = test_data.drop('COST', axis=1)
y_test = test_data['COST']

# 训练GBDT模型并使用GridSearchCV寻找最佳参数
param_grid = {
    'n_estimators': [140],
    'max_depth': [9],
    'learning_rate': [0.22],
    'min_samples_split': [4]

#     'n_estimators': list(np.arange(80, 150, 20)),
#     'max_depth': list(np.arange(5, 15, 1)),
#     'learning_rate': list(np.arange(0.18, 0.3, 0.02)),
#     'min_samples_split': list(np.arange(3, 6, 1))
}

model = GradientBoostingRegressor()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_mae = -grid_search.best_score_

print("Best Parameters:", best_params)
print("Best MAE:", best_mae)

# 训练最佳模型
best_model = GradientBoostingRegressor(**best_params)
best_model.fit(X_train, y_train)

# 用训练数据测试模型
y_train_pred = best_model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
corr_coef_train = np.corrcoef(y_train, y_train_pred)[0, 1]
pe_train=calculate_custom_pe(y_train, y_train_pred)
MAPE_train=calculate_mape(y_train, y_train_pred)
print("训练数据评估结果:")
print("均方根误差 (RMSE):", rmse_train)
print("平均绝对误差 (MAE):", mae_train)
print("决定系数 (R2 Score):", r2_train)
print("相关系数:", corr_coef_train)
print("pe:",pe_train)
print("MAPE",MAPE_train)
# 用测试数据测试模型
y_test_pred = best_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
corr_coef_test = np.corrcoef(y_test, y_test_pred)[0, 1]
pe_test=calculate_custom_pe(y_test, y_test_pred)
MAPE_test=calculate_mape(y_test, y_test_pred)
print("\n测试数据评估结果:")
print("均方根误差 (RMSE):", rmse_test)
print("平均绝对误差 (MAE):", mae_test)
print("决定系数 (R2 Score):", r2_test)
print("相关系数:", corr_coef_test)
print("pe:",pe_test)
print("MAPE",MAPE_test)

# 设置全局字体大小
plt.rcParams.update({'font.size': 20})  # 设置字体大小为14
# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, color='red', marker='o', label='Train')
plt.scatter(y_test, y_test_pred, color='green', marker='s', label='Test')
min_value = min(min(y_train), min(y_test), min(y_train_pred), min(y_test_pred))
max_value = max(max(y_train), max(y_test), max(y_train_pred), max(y_test_pred))
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


'缩短运行时间SHAP部分依赖图可暂时省略'
# 使用SHAP分析
explainer = shap.Explainer(best_model)
shap_values = explainer(X_train)
# 输出总体特征图
shap.summary_plot(shap_values, X_train, plot_type="bar")
# 输出特征的Shap值热力图
shap.summary_plot(shap_values, X_train)

# 画出各Shapvalue的各特征部分依赖图
explainer2 = shap.Explainer(best_model)
shap_values2 = explainer2.shap_values(X_train)
# 绘制每个特征的SHAP依赖图
for feature_idx in range(len(X_train.columns)):
    shap.dependence_plot(feature_idx, shap_values2, X_train, interaction_index=None)
plt.show()

# 根据shap值对特征进行排序
feature_importance = np.abs(shap_values.values).mean(0)
sorted_feature_names = X_train.columns[np.argsort(feature_importance)[::-1]]
print("Sorted Feature Names:", sorted_feature_names)

"节省运行时间直接输入SHAP排序结果"
sorted_feature_names=['Qin', 'CIout', 'CIin', 'RECOVERY', 'HARDout', 'SSin', 'SSout','HARDin', 'CODout', 'LEVEL',
                      'NH3Nout', 'CODin', 'LASout', 'YEAR', 'LASin', 'NH3Nin', 'CHROMin', 'CHROMout']

#存储重新选择参数后的R2值
r2_train_sorted_list=[]
r2_test_sorted_list=[]






"从一个参数到所有参数重新建模分析，得到最佳参数量后直接运行最佳参数量"
for i in range(9,len(sorted_feature_names)+1):
    # 读取训练数据和测试数据：
    X_train_sorted = X_train[sorted_feature_names[:i]]
    print("\n",i,'X_train_sorted:')
    print(X_train_sorted)
    y_train_sorted = y_train

    X_test_sorted = X_test[sorted_feature_names[:i]]
    y_test_sorted = y_test

    param_grid_sorted = {
        'n_estimators': [200],
        'max_depth': [7],
        'learning_rate': [0.22],
        'min_samples_split': [4]

        # 'n_estimators': list(np.arange(80, 150, 20)),
        # 'max_depth': list(np.arange(5, 15, 1)),
        # 'learning_rate': list(np.arange(0.18, 0.3, 0.02)),
        # 'min_samples_split': list(np.arange(3, 6, 1))
    }

    model_sorted = GradientBoostingRegressor()
    grid_search_sorted = GridSearchCV(model_sorted, param_grid_sorted, cv=5, scoring='neg_mean_absolute_error')
    grid_search_sorted.fit(X_train_sorted, y_train_sorted)

    best_params_sorted = grid_search_sorted.best_params_
    best_mae_sorted = -grid_search_sorted.best_score_

    print(i,"Best Parameters_sorted:", best_params_sorted)
    print(i,"Best MAE_sorted:", best_mae_sorted)

    # 训练最佳模型
    best_model_sorted = GradientBoostingRegressor(**best_params_sorted)
    best_model_sorted.fit(X_train_sorted, y_train_sorted)




    # 在训练数据上进行预测
    y_train_pred_sorted = best_model_sorted.predict(X_train_sorted)

    # 在测试数据上进行预测
    # 计算训练数据的均方根误差、平均绝对误差和调整的决定系数
    rmse_train_sorted = np.sqrt(mean_squared_error(y_train_sorted, y_train_pred_sorted))
    mae_train_sorted = mean_absolute_error(y_train_sorted, y_train_pred_sorted)
    r2_train_sorted = r2_score(y_train_sorted, y_train_pred_sorted)
    corr_coef_train_sorted = np.corrcoef(y_train_sorted, y_train_pred_sorted)[0, 1]
    pe_train_sorted = calculate_custom_pe(y_train_sorted, y_train_pred_sorted)
    MAPE_train_sorted = calculate_mape(y_train_sorted, y_train_pred_sorted)
    print("\n", i, "个参数输入训练数据评估结果:")
    print("均方根误差 (RMSE):", rmse_train_sorted)
    print("平均绝对误差 (MAE):", mae_train_sorted)
    print("决定系数 (R2 Score):", r2_train_sorted)
    print("相关系数:", corr_coef_train_sorted)
    print("pe_train_sorted:", pe_train_sorted)
    print("MAPE_train_sorted",MAPE_train_sorted)

    r2_train_sorted_list.append(r2_train_sorted)

    # 在测试数据上进行预测
    y_test_pred_sorted = best_model_sorted.predict(X_test_sorted)

    # 计算测试数据的r2
    # 计算测试数据的均方根误差、平均绝对误差和调整的决定系数
    rmse_test_sorted = np.sqrt(mean_squared_error(y_test_sorted, y_test_pred_sorted))
    mae_test_sorted = mean_absolute_error(y_test_sorted, y_test_pred_sorted)
    r2_test_sorted = r2_score(y_test_sorted, y_test_pred_sorted)
    corr_coef_test_sorted = np.corrcoef(y_test_sorted, y_test_pred_sorted)[0, 1]
    pe_test_sorted = calculate_custom_pe(y_test_sorted, y_test_pred_sorted)
    MAPE_test_sorted = calculate_mape(y_test_sorted, y_test_pred_sorted)
    print("\n", i, "个参数输入测试数据评估结果:")
    print("均方根误差 (RMSE):", rmse_test_sorted)
    print("平均绝对误差 (MAE):", mae_test_sorted)
    print("决定系数 (R2 Score):", r2_test_sorted)
    print("相关系数:", corr_coef_test_sorted)
    print("pe_test_sorted:", pe_test_sorted)
    print("MAPE_test_sorted",MAPE_test_sorted)

    r2_test_sorted_list.append(r2_test_sorted)


    # 绘制最佳参数数量时的真实值与预测值的散点图与残差分布图
    '此段输入需要进一步分析的参数数量'
    if i==9: #数字根据需要修改


        print("\n",i,"个参数输入训练数据评估结果:")
        print("均方根误差 (RMSE):", rmse_train_sorted)
        print("平均绝对误差 (MAE):", mae_train_sorted)
        print("决定系数 (R2 Score):", r2_train_sorted)
        print("相关系数:", corr_coef_train_sorted)

        print("\n",i,"个参数输入测试数据评估结果:")
        print("均方根误差 (RMSE):", rmse_test_sorted)
        print("平均绝对误差 (MAE):", mae_test_sorted)
        print("决定系数 (R2 Score):", r2_test_sorted)
        print("相关系数:", corr_coef_test_sorted)

        # 设置全局字体大小
        plt.rcParams.update({'font.size': 20})  # 设置字体大小为14
        # 绘制最佳参数数量时的真实值与预测值的散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(y_train_sorted, y_train_pred_sorted, color='red', marker='o', label='Train')
        plt.scatter(y_test_sorted, y_test_pred_sorted, color='green', marker='s', label='Test')
        min_value = min(min(y_train_sorted), min(y_test_sorted), min(y_train_pred_sorted), min(y_test_pred_sorted))
        max_value = max(max(y_train_sorted), max(y_test_sorted), max(y_train_pred_sorted), max(y_test_pred_sorted))
        plt.xlim(min_value, max_value)
        plt.ylim(min_value, max_value)
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
        all_predictions = best_model_sorted.predict(all_data)
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