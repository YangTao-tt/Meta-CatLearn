import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import optuna
import random

# seed
seed = 24
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# cuda
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


df = pd.read_csv("data.csv").astype(float)
catalyst_names = df.iloc[:, 0].values
X = df.drop(columns=["Barrier", df.columns[0]]).astype(float).values
y = df["Barrier"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# PyTorch
class DNNModel(nn.Module):
    def __init__(self, input_dim, units1, dropout1, units2, dropout2):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, units1)
        self.bn1 = nn.BatchNorm1d(units1)
        self.drop1 = nn.Dropout(dropout1)

        self.fc2 = nn.Linear(units1, units2)
        self.bn2 = nn.BatchNorm1d(units2)
        self.drop2 = nn.Dropout(dropout2)

        self.fc3 = nn.Linear(units2, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.drop4 = nn.Dropout(0.1)

        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = self.drop1(self.bn1(F.leaky_relu(self.fc1(x), 0.1)))
        x = self.drop2(self.bn2(F.leaky_relu(self.fc2(x), 0.1)))
        x = self.drop3(self.bn3(F.leaky_relu(self.fc3(x), 0.1)))
        x = self.drop4(self.bn4(F.leaky_relu(self.fc4(x), 0.1)))
        return self.out(x).squeeze(1)

# Optuna
def dnn_objective(trial):
    params = {
        "units1": trial.suggest_int("units1", 256, 1024),
        "dropout1": trial.suggest_float("dropout1", 0.2, 0.5),
        "units2": trial.suggest_int("units2", 128, 512),
        "dropout2": trial.suggest_float("dropout2", 0.2, 0.5),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        model = DNNModel(X.shape[1], params['units1'], params['dropout1'],
                         params['units2'], params['dropout2']).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience, counter = 20, 0

        for epoch in range(500):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = model(xb)
                    val_losses.append(criterion(pred, yb).item())
            val_loss = np.mean(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                break

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val.to(DEVICE)).cpu().numpy()
        rmse = np.sqrt(mean_squared_error(y_val.numpy(), y_val_pred))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

study = optuna.create_study(direction="minimize")
study.optimize(dnn_objective, n_trials=50)

print("Best Params:", study.best_params)
print("Best RMSE:", study.best_value)


X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(
    X, y, catalyst_names, test_size=0.2, random_state=seed
)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(X_test, y_test)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = DNNModel(X.shape[1], **{k: v for k, v in study.best_params.items() if k != 'lr'}).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=study.best_params['lr'])
criterion = nn.MSELoss()

train_losses = []
val_losses = []

for epoch in range(500):
    model.train()
    batch_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(np.mean(batch_losses))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            val_loss += criterion(pred, yb).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

model.eval()
y_train_pred = model(X_train.to(DEVICE)).cpu().detach().numpy()
y_test_pred = model(X_test.to(DEVICE)).cpu().detach().numpy()

print("\nTraining Set Performance:")
print(f"MAE:  {mean_absolute_error(y_train.numpy(), y_train_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train.numpy(), y_train_pred)):.4f}")
print(f"R^2:  {r2_score(y_train.numpy(), y_train_pred):.4f}")

print("\nTest Set Performance:")
print(f"MAE:  {mean_absolute_error(y_test.numpy(), y_test_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test.numpy(), y_test_pred)):.4f}")
print(f"R^2:  {r2_score(y_test.numpy(), y_test_pred):.4f}")

torch.save(model.state_dict(), "DNN.pth")

#
# train_predictions_df = pd.DataFrame({
#     "Catalyst": name_train,
#     "True": y_train,
#     "Predicted": y_train_pred
# })
# train_predictions_df.to_csv("train_predictions.csv", index=False)
# print("Training set predictions saved to train_predictions.csv")
#
# test_predictions_df = pd.DataFrame({
#     "Catalyst": name_test,
#     "True": y_test,
#     "Predicted": y_test_pred
# })
# test_predictions_df.to_csv("test_predictions.csv", index=False)


#
# plt.figure(figsize=(8, 5))
# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("MSE Loss")
# plt.title("DNN Training and Validation Loss Curve (PyTorch)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("loss_curve_optimized_pytorch.png", dpi=600)
# plt.show()















# # SHAP 分析
# def model_predict(x_numpy):
#     x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(DEVICE)
#     with torch.no_grad():
#         return model(x_tensor).cpu().numpy()
#
# # 使用所有数据点作为背景数据
# explainer = shap.KernelExplainer(model_predict, X.numpy())
# shap_values = explainer.shap_values(X.numpy())
#
# # 计算特征重要性
# feature_names = df.drop(columns=["Barrier", df.columns[0]]).columns
# mean_abs_shap = np.abs(shap_values).mean(axis=0)
# feature_importance = pd.DataFrame({
#     "Feature": feature_names,
#     "Mean_ABS_SHAP": mean_abs_shap
# }).sort_values("Mean_ABS_SHAP", ascending=False)
#
# # 打印前20
# print("\nTop 20 Important Features (SHAP):")
# print(feature_importance.head(20))
#
# # 绘制 SHAP 条形图
# plt.figure(figsize=(8, 6))
# plt.barh(feature_importance["Feature"][:20][::-1],
#          feature_importance["Mean_ABS_SHAP"][:20][::-1],
#          color="skyblue")
# plt.xlabel("Mean |SHAP Value|")
# plt.title("Top 20 Feature Importances by SHAP")
# plt.tight_layout()
# plt.savefig("shap_feature_importance_top20.png", dpi=300)
# plt.show()
#
# # SHAP Summary Plot
# top20_features = feature_importance["Feature"].values[:20]
# top20_indices = [feature_names.get_loc(f) for f in top20_features]
#
# shap.summary_plot(shap_values[:, top20_indices],
#                   X.numpy()[:, top20_indices],
#                   feature_names=top20_features)
#
# # 确保目录存在
# os.makedirs("shap_plots", exist_ok=True)
#
# # 保存 summary_plot
# plt.figure()
# shap.summary_plot(shap_values[:, top20_indices],
#                   X.numpy()[:, top20_indices],
#                   feature_names=top20_features,
#                   show=False)
# plt.tight_layout()
# plt.savefig("shap_plots/shap_summary_plot_top20.png", dpi=300)
# plt.close()
#
# # 保存 dependence plots（前5个）
# for i, feature in enumerate(top20_features[:5]):
#     idx = feature_names.get_loc(feature)
#     shap.dependence_plot(
#         idx, shap_values, X.numpy(),
#         feature_names=feature_names,
#         show=False
#     )
#     plt.tight_layout()
#     plt.savefig(f"shap_plots/depplot_{i+1}_{feature}.png", dpi=300)
#     plt.close()





# import os
# import random
# import numpy as np
# import pandas as pd
# import optuna
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.regularizers import l2
# import matplotlib.pyplot as plt
#
# # 固定随机种子
# seed = 42
# np.random.seed(seed)
# tf.random.set_seed(seed)
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
#
# # === 数据加载与标准化 ===
# df = pd.read_csv("data_scaled.csv").astype(float)
# catalyst_names = df.iloc[:, 0].values
# X = df.drop(columns=["Barrier", df.columns[0]]).astype(float).values
# y = df["Barrier"].values
#
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
#
# # === 构建改进版 DNN 模型 ===
# def create_dnn_model(params):
#     model = Sequential()
#     model.add(Dense(params["units1"], input_dim=X.shape[1], kernel_regularizer=l2(1e-4)))
#     model.add(LeakyReLU(alpha=0.1))
#     model.add(BatchNormalization())
#     model.add(Dropout(params["dropout1"]))
#
#     model.add(Dense(params["units2"], kernel_regularizer=l2(1e-4)))
#     model.add(LeakyReLU(alpha=0.1))
#     model.add(BatchNormalization())
#     model.add(Dropout(params["dropout2"]))
#
#     model.add(Dense(32, kernel_regularizer=l2(1e-4)))
#     model.add(LeakyReLU(alpha=0.1))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.2))
#
#     model.add(Dense(16, kernel_regularizer=l2(1e-4)))
#     model.add(LeakyReLU(alpha=0.1))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.1))
#
#     model.add(Dense(1))
#     model.compile(optimizer=Adam(learning_rate=params["lr"], amsgrad=True), loss="mean_squared_error")
#     return model
#
# # === Optuna 超参数优化目标函数 ===
# def dnn_objective(trial):
#     params = {
#         "units1": trial.suggest_int("units1", 256, 1024),
#         "dropout1": trial.suggest_float("dropout1", 0.2, 0.5),
#         "units2": trial.suggest_int("units2", 128, 512),
#         "dropout2": trial.suggest_float("dropout2", 0.2, 0.5),
#         "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True)
#     }
#
#     kf = KFold(n_splits=5, shuffle=True, random_state=seed)
#     rmse_scores = []
#
#     for train_idx, val_idx in kf.split(X):
#         X_train_cv, X_val_cv = X[train_idx], X[val_idx]
#         y_train_cv, y_val_cv = y[train_idx], y[val_idx]
#
#         model = create_dnn_model(params)
#         model.fit(
#             X_train_cv, y_train_cv,
#             validation_data=(X_val_cv, y_val_cv),
#             epochs=200,
#             batch_size=64,
#             verbose=0,
#             callbacks=[
#                 EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
#             ]
#         )
#
#         y_val_pred = model.predict(X_val_cv).flatten()
#         rmse = mean_squared_error(y_val_cv, y_val_pred, squared=False)
#         rmse_scores.append(rmse)
#
#     return np.mean(rmse_scores)
#
# # === 启动 Optuna 优化 ===
# study_dnn = optuna.create_study(direction="minimize")
# study_dnn.optimize(dnn_objective, n_trials=50)
#
# print("Best DNN Params:", study_dnn.best_params)
# print("Best DNN CV-RMSE:", study_dnn.best_value)
#
# # === 最终训练 & 测试评估 ===
# X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(
#     X, y, catalyst_names, test_size=0.2, random_state=seed
# )
#
# dnn_model = create_dnn_model(study_dnn.best_params)
# history = dnn_model.fit(
#     X_train, y_train,
#     validation_split=0.1,
#     epochs=200,
#     batch_size=64,
#     verbose=0,
#     callbacks=[
#         EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
#     ]
# )
#
# # === 训练集评估 ===
# y_train_pred = dnn_model.predict(X_train).flatten()
# print("\nTraining Set Performance:")
# print(f"MAE:  {mean_absolute_error(y_train, y_train_pred):.4f}")
# print(f"RMSE: {mean_squared_error(y_train, y_train_pred, squared=False):.4f}")
# print(f"R²:   {r2_score(y_train, y_train_pred):.4f}")
#
# # === 测试集评估 ===
# y_test_pred = dnn_model.predict(X_test).flatten()
# print("\nTest Set Performance:")
# print(f"MAE:  {mean_absolute_error(y_test, y_test_pred):.4f}")
# print(f"RMSE: {mean_squared_error(y_test, y_test_pred, squared=False):.4f}")
# print(f"R²:   {r2_score(y_test, y_test_pred):.4f}")
#
# # === 保存模型与结果 ===
# dnn_model.save("DNN1.h5")
# print("\nModel saved to DNN1.h5")
#
# pd.DataFrame({
#     "Catalyst": name_train,
#     "True": y_train,
#     "Predicted": y_train_pred
# }).to_csv("train_predictions1.csv", index=False)
#
# pd.DataFrame({
#     "Catalyst": name_test,
#     "True": y_test,
#     "Predicted": y_test_pred
# }).to_csv("test_predictions1.csv", index=False)
#
# # === 绘制损失曲线 ===
# plt.figure(figsize=(8, 5))
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
#
# # plt.axvline(x=best_iter, color='r', linestyle='--', label=f"Early Stop @ {best_iter}")
#
#
# plt.xlabel("Epoch")
# plt.ylabel("MSE Loss")
# plt.title("DNN Training and Validation Loss Curve (Optimized)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("loss_curve_optimized.png", dpi=600)
# plt.show()
#
