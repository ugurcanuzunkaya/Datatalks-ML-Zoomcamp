import pickle
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score

# Parameters
output_file = "models/model.bin"
test_size = 0.2
random_state = 42


def train() -> None:
    """
    Main training pipeline:
    1. Loads dataset.
    2. Splits data into train/test.
    3. Vectorizes features.
    4. Trains an XGBoost model.
    5. Logs metrics and artifacts to MLflow.
    6. Saves the final model locally.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("stress_level_prediction")

    with mlflow.start_run():
        print("Loading data...")
        # Assuming script is run from project root, so path is valid
        # If dataset was moved, we need to adjust. We assume it's still in root for now.
        if os.path.exists("StressLevelDataset.csv"):
            data_path = "StressLevelDataset.csv"
        else:
            raise FileNotFoundError("StressLevelDataset.csv not found")

        df = pd.read_csv(data_path)

        df_full_train, df_test = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        y_train = df_full_train.stress_level.values
        y_test = df_test.stress_level.values

        del df_full_train["stress_level"]
        del df_test["stress_level"]

        print("Vectorizing data...")
        dv = DictVectorizer(sparse=False)
        train_dicts = df_full_train.to_dict(orient="records")
        X_train = dv.fit_transform(train_dicts)

        test_dicts = df_test.to_dict(orient="records")
        X_test = dv.transform(test_dicts)

        # XGBoost Params
        xgb_params = {
            "eta": 0.3,
            "max_depth": 6,
            "min_child_weight": 1,
            "objective": "multi:softmax",
            "num_class": 3,
            "nthread": 8,
            "seed": 42,
            "verbosity": 1,
        }

        mlflow.log_params(xgb_params)

        print("Training model...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(xgb_params, dtrain, num_boost_round=100)

        y_pred = model.predict(dtest)
        acc = accuracy_score(y_test, y_pred)
        print(f"Validation Accuracy: {acc:.4f}")
        mlflow.log_metric("accuracy", acc)

        # Save model locally
        print(f"Saving model to {output_file}...")
        os.makedirs("models", exist_ok=True)
        with open(output_file, "wb") as f_out:
            pickle.dump((dv, model), f_out)

        # Log model artifact to MLflow
        mlflow.log_artifact(output_file)

        print("Done.")


if __name__ == "__main__":
    train()
