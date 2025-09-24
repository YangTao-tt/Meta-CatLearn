import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



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


def xgb_objective(trial):
    params = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1
    }

    model = XGBRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    return -scores.mean()

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(xgb_objective, n_trials=50)

print("Best XGBoost Params:", study_xgb.best_params)
print("Best XGBoost CV-RMSE:", study_xgb.best_value)


# XGBRegressor
best_params = study_xgb.best_params
best_params.update({
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 1000,
    "verbose": 0,
    "random_state": 42
})



xgb_model = XGBRegressor(**best_params)
xgb_model.fit(X_train, y_train)


y_train_pred = xgb_model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)


y_test_pred = xgb_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)


print(f"\nTraining Set - MAE:  {train_mae:.4f}")
print(f"Training Set - RMSE: {train_rmse:.4f}")
print(f"Training Set - R²:   {train_r2:.4f}")


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

import joblib
joblib.dump(xgb_model, './XGB.pkl')


