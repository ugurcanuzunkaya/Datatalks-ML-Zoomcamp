import pickle
import xgboost as xgb
import os
from src.models import StressInput, PredictionOutput

STRESS_LEVELS = {0: "Low", 1: "Medium", 2: "High"}


class Predictor:
    """
    Class to handle model loading and prediction logic for stress level classification.
    """

    def __init__(self, model_path: str):
        """
        Initialize the predictor by loading the model and vectorizer from the given path.

        Args:
            model_path (str): Path to the pickled model/vectorizer file.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        with open(model_path, "rb") as f:
            self.dv, self.model = pickle.load(f)

    def predict_one(self, data: StressInput) -> PredictionOutput:
        """
        Predict stress level for a single input.

        Args:
            data (StressInput): Input features validated by Pydantic.

        Returns:
            PredictionOutput: The predicted stress level code and label.
        """
        # Convert Pydantic model to dict
        data_dict = data.model_dump()

        # Transform features using the loaded DictVectorizer
        X = self.dv.transform([data_dict])
        dmatrix = xgb.DMatrix(X)

        # Predict using XGBoost model
        y_pred = self.model.predict(dmatrix)
        prediction = int(y_pred[0])

        stress_label = STRESS_LEVELS.get(prediction, "Unknown")

        return PredictionOutput(stress_level_code=prediction, stress_level=stress_label)
