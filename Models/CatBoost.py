import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import optuna
import matplotlib.pyplot as plt
import joblib


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


# Optuna
def cat_objective(trial):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "verbose": 0,
        "random_state": 42
    }
    model = CatBoostRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    return -scores.mean()

study_cat = optuna.create_study(direction="minimize")
study_cat.optimize(cat_objective, n_trials=50)

print("Best CatBoost Params:", study_cat.best_params)
print("Best CatBoost CV-RMSE:", study_cat.best_value)



best_params = study_cat.best_params
best_params.update({
    "iterations": 1000,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "verbose": 0,
    "random_state": 42
})



cat_model = CatBoostRegressor(**best_params)


eval_set = Pool(X_test, y_test)


cat_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=20,
    use_best_model=True,
    verbose=50
)

# #
# plt.figure(figsize=(8, 5))
# plt.plot(cat_model.get_evals_result()['learn']['RMSE'], label='Train RMSE')
# plt.plot(cat_model.get_evals_result()['validation']['RMSE'], label='Validation RMSE')
# plt.xlabel('Iterations')
# plt.ylabel('RMSE')
# plt.title('CatBoost Training and Validation Loss')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("CAT_loss_curve.png", dpi=600)
# plt.show()


y_train_pred = cat_model.predict(X_train)
y_test_pred = cat_model.predict(X_test)


train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)


test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nTraining Set - MAE:  {train_mae:.4f}")
print(f"Training Set - RMSE: {train_rmse:.4f}")
print(f"Training Set - R²:   {train_r2:.4f}")

print(f"\nTest Set - MAE:  {test_mae:.4f}")
print(f"Test Set - RMSE: {test_rmse:.4f}")
print(f"Test Set - R²:   {test_r2:.4f}")

# save
joblib.dump(cat_model, './CatBoost.pkl')


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





# all_y = pd.concat([y_train, y_test])
#
# # 4. fig
# plt.figure(figsize=(6, 6))
#
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['axes.labelweight'] = 'bold'
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
# plt.title("CatBoost",fontsize=16, fontweight='bold')
# plt.legend(fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("CatBoost-true_vs_pred.png", dpi=600, bbox_inches='tight')








