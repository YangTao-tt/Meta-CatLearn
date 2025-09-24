import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 数据加载和预处理
df = pd.read_csv("data_scaled.csv").astype(float)
catalyst_names = df.iloc[:, 0].values
X = df.drop(columns=["Barrier"]).values
y = df["Barrier"].values

X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(
    X, y, catalyst_names, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义 Random Forest 的目标函数
def rf_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),  # 修改这里
        "random_state": 42
    }

    model = RandomForestRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    return -scores.mean()

# 使用 Optuna 进行超参数优化
study_rf = optuna.create_study(direction="minimize")
study_rf.optimize(rf_objective, n_trials=50)

print("Best Random Forest Params:", study_rf.best_params)
print("Best Random Forest CV-RMSE:", study_rf.best_value)

# 使用最佳参数重新训练 Random Forest 模型
best_params = study_rf.best_params

# 训练最终模型
rf_model = RandomForestRegressor(**best_params)
rf_model.fit(X_train, y_train)

# 评估训练集
y_train_pred = rf_model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# 评估测试集
y_test_pred = rf_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 输出训练集的性能指标
print(f"\nTraining Set - MAE:  {train_mae:.4f}")
print(f"Training Set - RMSE: {train_rmse:.4f}")
print(f"Training Set - R²:   {train_r2:.4f}")

# 输出测试集的性能指标
print(f"\nTest Set - MAE:  {test_mae:.4f}")
print(f"Test Set - RMSE: {test_rmse:.4f}")
print(f"Test Set - R²:   {test_r2:.4f}")

train_predictions_df = pd.DataFrame({
    "Catalyst": name_train,
    "True": y_train,
    "Predicted": y_train_pred
})
train_predictions_df.to_csv("train_predictions.csv", index=False)
print("Training set predictions saved to train_predictions.csv")

test_predictions_df = pd.DataFrame({
    "Catalyst": name_test,
    "True": y_test,
    "Predicted": y_test_pred
})
test_predictions_df.to_csv("test_predictions.csv", index=False)


# 保存模型
import joblib
joblib.dump(rf_model, './RF.pkl')





#
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import optuna
# import matplotlib.pyplot as plt
# import joblib
#
# # 数据加载和预处理
# df = pd.read_csv("data_scaled.csv").astype(float)
# catalyst_names = df.iloc[:, 0].values
# X = df.drop(columns=["Barrier"]).values
# y = df["Barrier"].values
#
# X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(
#     X, y, catalyst_names, test_size=0.2, random_state=42
# )
#
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# # 定义 Random Forest 的目标函数
# def rf_objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 50, 500),
#         "max_depth": trial.suggest_int("max_depth", 3, 20),
#         "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
#         "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
#         "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
#         "random_state": 42,
#         "n_jobs": -1
#     }
#
#     model = RandomForestRegressor(**params)
#     from sklearn.model_selection import KFold, cross_val_score
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
#     return -scores.mean()
#
# # Optuna搜索
# study_rf = optuna.create_study(direction="minimize")
# study_rf.optimize(rf_objective, n_trials=50)
#
# print("Best Random Forest Params:", study_rf.best_params)
# print("Best Random Forest CV-RMSE:", study_rf.best_value)
#
# # 用最优参数模拟 Early Stopping
# best_params = study_rf.best_params
#
#
# # 设置 early stopping 参数
# early_stop_rounds = 10
# max_trees = 200
#
# train_rmse_list = []
# val_rmse_list = []
#
# best_val_rmse = float("inf")
# best_iter = 0
# no_improve_rounds = 0
#
# rf_model = RandomForestRegressor(
#     warm_start=True,
#     oob_score=False,
#     **{k: v for k, v in best_params.items() if k != "n_estimators"}
# )
#
# for n_trees in range(1, max_trees + 1):
#     rf_model.set_params(n_estimators=n_trees)
#     rf_model.fit(X_train, y_train)
#
#     y_train_pred = rf_model.predict(X_train)
#     y_val_pred = rf_model.predict(X_val)
#
#     train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
#     val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
#
#     train_rmse_list.append(train_rmse)
#     val_rmse_list.append(val_rmse)
#
#     if val_rmse < best_val_rmse - 1e-6:
#         best_val_rmse = val_rmse
#         best_iter = n_trees
#         no_improve_rounds = 0
#     else:
#         no_improve_rounds += 1
#
#     if no_improve_rounds >= early_stop_rounds:
#         print(f"\nEarly stopping at {n_trees} trees, best at {best_iter} with RMSE = {best_val_rmse:.4f}")
#         break
#
# # 可视化训练和验证 RMSE
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(train_rmse_list) + 1), train_rmse_list, label='Train RMSE')
# plt.plot(range(1, len(val_rmse_list) + 1), val_rmse_list, label='Validation RMSE')
# plt.axvline(x=best_iter, color='r', linestyle='--', label=f'Early Stop @ {best_iter}')
# plt.xlabel('Number of Trees')
# plt.ylabel('RMSE')
# plt.title('Random Forest Training & Validation Loss')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("RF_loss_curve.png", dpi=600)
# plt.show()
#
# # 最终模型训练（使用最佳迭代次数）
# final_rf_model = RandomForestRegressor(
#     n_estimators=best_iter,
#     **{k: v for k, v in best_params.items() if k != "n_estimators"}
# )
# final_rf_model.fit(X_train, y_train)
#
# # 训练集评估
# y_train_pred = final_rf_model.predict(X_train)
# train_mae = mean_absolute_error(y_train, y_train_pred)
# train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
# train_r2 = r2_score(y_train, y_train_pred)
#
# # 验证集评估
# y_val_pred = final_rf_model.predict(X_val)
# val_mae = mean_absolute_error(y_val, y_val_pred)
# val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
# val_r2 = r2_score(y_val, y_val_pred)
#
# print(f"\nTraining Set - MAE:  {train_mae:.4f}")
# print(f"Training Set - RMSE: {train_rmse:.4f}")
# print(f"Training Set - R²:   {train_r2:.4f}")
#
# print(f"\nValidation Set - MAE:  {val_mae:.4f}")
# print(f"Validation Set - RMSE: {val_rmse:.4f}")
# print(f"Validation Set - R²:   {val_r2:.4f}")
#
# # 保存模型
# joblib.dump(final_rf_model, "./RF.pkl")
#
#
# all_y = pd.concat([y_train, y_test])
#
# # 4. 绘制图形
# plt.figure(figsize=(6, 6))
#
# plt.rcParams['font.weight'] = 'bold'  # 全局加粗
# plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
# plt.rcParams['axes.titleweight'] = 'bold'
#
# plt.scatter(y_train, y_train_pred, alpha=0.6, color='green', label='Train Set')
# plt.scatter(y_test, y_test_pred, alpha=0.6, color='blue', label='Test Set')
#
# plt.plot([all_y.min(), all_y.max()],
#          [all_y.min(), all_y.max()],
#              'r--', linewidth=1, label='Perfect Prediction')
#
#
# plt.xlabel("True Barrier",fontsize=14, fontweight='bold')
# plt.ylabel("Predicted Barrier",fontsize=14, fontweight='bold')
#
# ax = plt.gca()
# ax.tick_params(axis='both',which='major',labelsize=14, width=2)
#
# plt.title("Random Forest",fontsize=16, fontweight='bold')
# plt.legend(fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("RF-true_vs_pred.png", dpi=600, bbox_inches='tight')
