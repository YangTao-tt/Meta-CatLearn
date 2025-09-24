import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


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


df_ti = pd.read_csv("data_ti_scaled.csv").astype(float)
catalyst_names = df_ti.iloc[:, 0].values
X_ti = df_ti.drop(columns=["Barrier", df_ti.columns[0]]).values
y_ti = df_ti["Barrier"].values


scaler = StandardScaler()
scaler.fit(X_ti)
X_ti_scaled = scaler.transform(X_ti)
X_ti_tensor = torch.tensor(X_ti_scaled, dtype=torch.float32).to(DEVICE)
y_ti_tensor = torch.tensor(y_ti, dtype=torch.float32)


# DNN3.pth(Zr/Hf)

best_params = {
    'units1': 352,
    'dropout1': 0.30577904668726064,
    'units2': 392,
    'dropout2': 0.23916654446282398
}

model = DNNModel(X_ti.shape[1], **best_params).to(DEVICE)
model.load_state_dict(torch.load("DNN3.pth"))
model.eval()


#Zero-shot

with torch.no_grad():
    y_pred = model(X_ti_tensor).cpu().numpy()

mae = mean_absolute_error(y_ti, y_pred)
rmse = np.sqrt(mean_squared_error(y_ti, y_pred))
r2 = r2_score(y_ti, y_pred)

print(f"\n[Zero-shot Prediction Results on Ti Catalysts]")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


results_df = pd.DataFrame({
    "Catalyst": catalyst_names,
    "True_Barrier": y_ti,
    "Predicted_Barrier": y_pred.flatten(),
    "Absolute_Error": np.abs(y_ti - y_pred.flatten())
})

results_df.to_csv("predictions_ti.csv", index=False, encoding="utf-8-sig")
print("predictions_ti.csv")

