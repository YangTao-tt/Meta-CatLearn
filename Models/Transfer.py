import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 固定随机种子 & 设备
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ========================================
# 1. 模型结构（保持一致）
# ========================================
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

# ========================================
# 2. 加载 Ti 数据
# ========================================
df_ti = pd.read_csv("data_ti_scaled.csv").astype(float)
catalyst_names = df_ti.iloc[:, 0].values
X_ti = df_ti.drop(columns=["Barrier", df_ti.columns[0]]).values
y_ti = df_ti["Barrier"].values

# 标准化（可替换为预训练模型的 scaler）
scaler = StandardScaler()
X_ti_scaled = scaler.fit_transform(X_ti)

X_ti_tensor = torch.tensor(X_ti_scaled, dtype=torch.float32)
y_ti_tensor = torch.tensor(y_ti, dtype=torch.float32)

# 构造完整数据集
full_dataset = TensorDataset(X_ti_tensor, y_ti_tensor)
train_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)

# ========================================
# 3. 加载预训练模型（冻结前两层）
# ========================================
best_params = {
    'units1': 842,
    'dropout1': 0.35352037426452376,
    'units2': 512,
    'dropout2': 0.357381939264172,
}

model = DNNModel(X_ti.shape[1], **best_params).to(DEVICE)
model.load_state_dict(torch.load("DNN1.pth"))

# 冻结前两层
for name, param in model.named_parameters():
    if "fc1" in name or "fc2" in name:
        param.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = nn.MSELoss()

# ========================================
# 4. 微调训练（全部数据）
# ========================================
train_losses = []

for epoch in range(1000):
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
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}: Train Loss = {train_loss:.4f}")

# ========================================
# 5. 模型在所有数据上评估
# ========================================
model.eval()
with torch.no_grad():
    y_pred = model(X_ti_tensor.to(DEVICE)).cpu().numpy()

mae = mean_absolute_error(y_ti, y_pred)
rmse = np.sqrt(mean_squared_error(y_ti, y_pred))
r2 = r2_score(y_ti, y_pred)

print(f"\n[Transfer Learning on All Ti Data]")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ========================================
# 6. 保存预测结果
# ========================================
results_df = pd.DataFrame({
    "Catalyst": catalyst_names,
    "True_Barrier": y_ti,
    "Predicted_Barrier": y_pred.flatten(),
    "Absolute_Error": np.abs(y_ti - y_pred.flatten())
})
results_df.to_csv("predictions_ti_transfer_all.csv", index=False, encoding="utf-8-sig")
print("✅ 所有Ti数据的预测结果已保存至 predictions_ti_transfer_all.csv")






# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import pandas as pd
# from torch.utils.data import DataLoader, TensorDataset, random_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
#
# # 固定随机种子 & 设备
# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", DEVICE)
#
# # ========================================
# # 1. 模型结构（保持一致）
# # ========================================
# class DNNModel(nn.Module):
#     def __init__(self, input_dim, units1, dropout1, units2, dropout2):
#         super(DNNModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim, units1)
#         self.bn1 = nn.BatchNorm1d(units1)
#         self.drop1 = nn.Dropout(dropout1)
#
#         self.fc2 = nn.Linear(units1, units2)
#         self.bn2 = nn.BatchNorm1d(units2)
#         self.drop2 = nn.Dropout(dropout2)
#
#         self.fc3 = nn.Linear(units2, 32)
#         self.bn3 = nn.BatchNorm1d(32)
#         self.drop3 = nn.Dropout(0.2)
#
#         self.fc4 = nn.Linear(32, 16)
#         self.bn4 = nn.BatchNorm1d(16)
#         self.drop4 = nn.Dropout(0.1)
#
#         self.out = nn.Linear(16, 1)
#
#     def forward(self, x):
#         x = self.drop1(self.bn1(F.leaky_relu(self.fc1(x), 0.1)))
#         x = self.drop2(self.bn2(F.leaky_relu(self.fc2(x), 0.1)))
#         x = self.drop3(self.bn3(F.leaky_relu(self.fc3(x), 0.1)))
#         x = self.drop4(self.bn4(F.leaky_relu(self.fc4(x), 0.1)))
#         return self.out(x).squeeze(1)
#
# # ========================================
# # 2. 加载 Ti 数据
# # ========================================
# df_ti = pd.read_csv("data_ti_scaled.csv").astype(float)
# catalyst_names = df_ti.iloc[:, 0].values
# X_ti = df_ti.drop(columns=["Barrier", df_ti.columns[0]]).values
# y_ti = df_ti["Barrier"].values
#
#
# scaler = StandardScaler()
# X_ti_scaled = scaler.fit_transform(X_ti)
#
# X_ti_tensor = torch.tensor(X_ti_scaled, dtype=torch.float32)
# y_ti_tensor = torch.tensor(y_ti, dtype=torch.float32)
#
# # 构造训练/测试集
# dataset = TensorDataset(X_ti_tensor, y_ti_tensor)
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
#
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
# # ========================================
# # 3. 加载预训练模型并冻结前两层
# # ========================================
# best_params = {
#     'units1': 842,
#     'dropout1': 0.35352037426452376,
#     'units2': 512,
#     'dropout2': 0.357381939264172,
# }
#
#
# model = DNNModel(X_ti.shape[1], **best_params).to(DEVICE)
# model.load_state_dict(torch.load("DNN1.pth"))
#
# # 冻结前两层参数（fc1, fc2）
# for name, param in model.named_parameters():
#     if "fc1" in name or "fc2" in name:
#         param.requires_grad = False
#
# # 定义优化器和损失函数（只更新可训练参数）
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# criterion = nn.MSELoss()
#
# # ========================================
# # 4. 微调训练
# # ========================================
# train_losses, val_losses = [], []
#
# for epoch in range(200):
#     model.train()
#     batch_losses = []
#     for xb, yb in train_loader:
#         xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#         optimizer.zero_grad()
#         pred = model(xb)
#         loss = criterion(pred, yb)
#         loss.backward()
#         optimizer.step()
#         batch_losses.append(loss.item())
#     train_losses.append(np.mean(batch_losses))
#
#     # 验证
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for xb, yb in test_loader:
#             xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#             pred = model(xb)
#             val_loss += criterion(pred, yb).item()
#     val_loss /= len(test_loader)
#     val_losses.append(val_loss)
#
# # ========================================
# # 5. 性能评估
# # ========================================
# model.eval()
# with torch.no_grad():
#     X_test_tensor = torch.stack([x[0] for x in test_dataset]).to(DEVICE)
#     y_test_tensor = torch.stack([x[1] for x in test_dataset])
#     y_pred = model(X_test_tensor).cpu().numpy()
#
# mae = mean_absolute_error(y_test_tensor, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test_tensor, y_pred))
# r2 = r2_score(y_test_tensor, y_pred)
#
# print(f"\n[Transfer Learning on Ti Dataset]")
# print(f"MAE  : {mae:.4f}")
# print(f"RMSE : {rmse:.4f}")
# print(f"R²   : {r2:.4f}")
#
# # ========================================
# # 6. 保存预测结果
# # ========================================
# results_df = pd.DataFrame({
#     "Catalyst": catalyst_names[-len(y_test_tensor):],  # 保证顺序一致
#     "True_Barrier": y_test_tensor.numpy(),
#     "Predicted_Barrier": y_pred.flatten(),
#     "Absolute_Error": np.abs(y_test_tensor.numpy() - y_pred.flatten())
# })
# results_df.to_csv("predictions_ti_transfer.csv", index=False, encoding="utf-8-sig")
# print("✅ 微调预测结果已保存至 predictions_ti_transfer.csv")
