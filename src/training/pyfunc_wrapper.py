import mlflow.pyfunc
import pandas as pd


class ChurnModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc adapter exposing churn probabilities and classifications."""
    def __init__(self, model, threshold=None):
        self.model = model
        self.threshold = threshold

    def predict(self, context, model_input: pd.DataFrame):
        probs = self.model.predict_proba(model_input)[:, 1]

        return probs