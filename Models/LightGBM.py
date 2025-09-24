import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

def lgb_objective(trial):
    params = {
        "n_estimators": 200,
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
    }

    model = LGBMRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    return -scores.mean()

study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(lgb_objective, n_trials=30)

print("Best LGBM Params:", study_lgb.best_params)
print("Best LGBM CV-RMSE:", study_lgb.best_value)


# CatBoost
best_params = study_lgb.best_params
best_params.update({
    "iterations": 1000,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "verbose": 0,
    "random_state": 42
})



lgb_model = LGBMRegressor(**best_params)
lgb_model.fit(X_train, y_train)



y_train_pred = lgb_model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)


y_test_pred = lgb_model.predict(X_test)
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
joblib.dump(lgb_model, './LGBM.pkl')








