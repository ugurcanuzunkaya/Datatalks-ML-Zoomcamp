import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pickle
import os


def build_pipeline(model_type, params):
    """
    Build a preprocessing + model pipeline.
    """
    numeric_features = ["age", "bmi", "children"]
    # Smoker and sex are binary, region is nominal
    categorical_features = ["sex", "smoker", "region"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    if model_type == "linear":
        model = LinearRegression(**params)
    elif model_type == "rf":
        model = RandomForestRegressor(**params, random_state=42)
    elif model_type == "xgb":
        model = XGBRegressor(**params, random_state=42)
    else:
        raise ValueError("Unknown model type")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def train(data_path: str):
    mlflow.set_experiment("insurance_costs_prediction")

    # Load Data
    df = pd.read_csv(data_path)
    X = df.drop(columns=["charges"])
    y = np.log1p(df["charges"])  # Log-transform target

    # Define models and simple hyperparameter grids (for demonstration)
    # In a real scenario, we'd loop over these grids more extensively
    configs = [
        ("linear", {}),
        ("linear", {"fit_intercept": False}),
        ("rf", {"n_estimators": 100, "max_depth": 10}),
        ("rf", {"n_estimators": 100, "max_depth": 20}),
        ("rf", {"n_estimators": 200, "max_depth": 20}),
        ("rf", {"n_estimators": 200, "max_depth": 10}),
        ("rf", {"n_estimators": 300, "max_depth": 10}),
        ("rf", {"n_estimators": 300, "max_depth": 20}),
        ("rf", {"n_estimators": 400, "max_depth": 10}),
        ("rf", {"n_estimators": 400, "max_depth": 20}),
        ("xgb", {"n_estimators": 100, "learning_rate": 0.1}),
        ("xgb", {"n_estimators": 100, "learning_rate": 0.05}),
        ("xgb", {"n_estimators": 100, "learning_rate": 0.01}),
        ("xgb", {"n_estimators": 200, "learning_rate": 0.1}),
        ("xgb", {"n_estimators": 200, "learning_rate": 0.05}),
        ("xgb", {"n_estimators": 200, "learning_rate": 0.01}),
        ("xgb", {"n_estimators": 300, "learning_rate": 0.1}),
        ("xgb", {"n_estimators": 300, "learning_rate": 0.05}),
        ("xgb", {"n_estimators": 300, "learning_rate": 0.01}),
        ("xgb", {"n_estimators": 400, "learning_rate": 0.1}),
        ("xgb", {"n_estimators": 400, "learning_rate": 0.05}),
        ("xgb", {"n_estimators": 400, "learning_rate": 0.01}),
    ]

    best_rmse = float("inf")
    best_pipeline = None
    best_config = None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_type, params in configs:
        with mlflow.start_run():
            pipeline = build_pipeline(model_type, params)

            # Log params
            mlflow.log_param("model_type", model_type)
            mlflow.log_params(params)

            # K-Fold CV
            # We calculate RMSE manually on the folded predictions to handle inverse transform properly
            rmses = []
            for train_index, val_index in kf.split(X, y):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                pipeline.fit(X_train, y_train)
                y_pred_log = pipeline.predict(X_val)

                # Inverse transform predictions and actuals to get error in $
                y_pred = np.expm1(y_pred_log)
                y_val_orig = np.expm1(y_val)

                rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred))
                rmses.append(rmse)

            avg_rmse = np.mean(rmses)

            # Log metrics
            mlflow.log_metric("cv_rmse", avg_rmse)
            print(f"Model: {model_type}, Params: {params}, CV RMSE: ${avg_rmse:.2f}")

            # Track best model
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_pipeline = pipeline
                best_config = (model_type, params)

            # Fit on full data for artifact logging (optional, usually we save the best only)
            pipeline.fit(X, y)
            mlflow.sklearn.log_model(pipeline, name="model")

    print(f"\nBest Model: {best_config[0]} with Params: {best_config[1]} (RMSE: {best_rmse})")

    # Retrain best pipeline on full dataset and save
    best_pipeline_final = build_pipeline(best_config[0], best_config[1])
    best_pipeline_final.fit(X, y)

    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_pipeline_final, f)
    print("Saved best model to models/best_model.pkl")


if __name__ == "__main__":
    train("insurance.csv")
