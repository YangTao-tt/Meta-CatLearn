import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import joblib


# 数据读取与预处理
df = pd.read_csv("data_scaled.csv").astype(float)
catalyst_names = df.iloc[:, 0].values
X = df.drop(columns=["Barrier"]).values
y = df["Barrier"].values


# 划分训练和测试集
X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(
    X, y, catalyst_names, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 超参数优化目标函数
def gradient_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42
    }

    model = GradientBoostingRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    return -scores.mean()

# 创建 Optuna 研究对象并优化
study_gradient = optuna.create_study(direction="minimize")
study_gradient.optimize(gradient_objective, n_trials=50)

print("Best Gradient Boosting Params:", study_gradient.best_params)
print("Best Gradient Boosting CV-RMSE:", study_gradient.best_value)

# 使用最佳参数重新训练模型
best_params = study_gradient.best_params

# 训练最终模型
gradient_model = GradientBoostingRegressor(**best_params)
gradient_model.fit(X_train, y_train)

# 在训练集上评估模型
y_train_pred = gradient_model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)

# 在测试集上评估模型
y_test_pred = gradient_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
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
joblib.dump(gradient_model, './GBR.pkl')



#1.散点图
# plt.figure(figsize=(6, 6))
# plt.scatter(y_test, y_test_pred, alpha=0.6, label='Test Set')
# plt.scatter(y_train, y_train_pred, alpha=0.6, label='Training Set')
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
# plt.xlabel("True Barrier")
# plt.ylabel("Predicted Barrier")
# plt.title("GBR-True vs Predicted Barrier")
# plt.legend()
# plt.tight_layout()
# plt.grid(True)
# plt.savefig("true_vs_pred.png", dpi=600)
# plt.show()
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
# plt.title("GradientBoosting",fontsize=16, fontweight='bold')
# plt.legend(fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("GBR-true_vs_pred.png", dpi=600, bbox_inches='tight')
#
#
#
#
# # 2.损失曲线图
# plt.figure(figsize=(6, 6))
# losses = gradient_model.loss
# plt.plot(range(1, len(losses) + 1), losses, label='Loss')
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.title("Loss Curve")
# plt.legend()
# plt.tight_layout()
# plt.grid(True)
# plt.savefig("loss_curve.png", dpi=600)
# plt.show()
#
# # 3.学习曲线图
# from sklearn.model_selection import learning_curve
#
# train_sizes, train_scores, test_scores = learning_curve(
#     gradient_model, X, y, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1,
#     train_sizes=np.linspace(0.1, 1.0, 10)
# )
#
# train_scores_mean = -np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = -np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.figure(figsize=(6, 6))
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
# plt.xlabel("Training examples")
# plt.ylabel("RMSE")
# plt.title("Learning Curve")
# plt.legend(loc="best")
# plt.grid(True)
# plt.savefig("learning_curve.png", dpi=600)
# plt.show()
#
# # SHAP 分析
# explainer = shap.Explainer(gradient_model, X_train)
# shap_values = explainer(X_test)
#
# # SHAP 总结图
# shap.summary_plot(shap_values, X_test, feature_names=df.columns[:-1], max_display=10)
# plt.savefig("shap_summary.png", dpi=600)
# plt.show()
#
# # SHAP 值图
# shap.plots.waterfall(shap_values[0], max_display=10)
# plt.savefig("shap_waterfall.png", dpi=600)
# plt.show()