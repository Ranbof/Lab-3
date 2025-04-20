import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib

import os

if not os.path.exists('taxi_features.csv') or not os.path.exists('taxi_target.csv'):
    raise FileNotFoundError("Required CSV files not found")

def scale_data(X, y):
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    
    X_scaled = scaler.fit_transform(X)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))
    
    return X_scaled, y_scaled, power_trans

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    X = pd.read_csv('taxi_features.csv')
    y = pd.read_csv('taxi_target.csv')

    X_scaled, y_scaled, power_trans = scale_data(X, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=42
    )
    
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2]
    }
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("taxi")
    
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=5)
        clf.fit(X_train, y_train.ravel())
        
        best_model = clf.best_estimator_
        
        y_pred = best_model.predict(X_val)
        y_pred_original = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        y_val_original = power_trans.inverse_transform(y_val)
        
        rmse, mae, r2 = eval_metrics(y_val_original, y_pred_original)
        
        mlflow.log_param("alpha", best_model.alpha)
        mlflow.log_param("l1_ratio", best_model.l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        joblib.dump(best_model, "taxi_model.joblib")
        
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2: {r2:.2f}")

        model_path = os.path.join("mlruns", mlflow.active_run().info.run_id, "artifacts", "model")
        with open("best_model.txt", "w") as f:
            f.write(model_path)
        mlflow.log_artifact("best_model.txt")
