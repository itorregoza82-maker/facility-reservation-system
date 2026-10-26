
import joblib
import pandas as pd
from pathlib import Path

from modules.data_processing import clean_model_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "trained_model.joblib"

_model = None

def load_model():
    """Load the saved model once and reuse it."""
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Saved model not found at: {MODEL_PATH}"
            )
        _model = joblib.load(MODEL_PATH)

    return _model

def predict_reservations(df):
    """Predict reservation outcomes for rows containing valid model features."""
    model = load_model()
    features = clean_model_features(df)

    predictions = model.predict(features)

    results = pd.DataFrame(index=df.index)
    results["Predicted_Outcome"] = [
        "Completed" if int(value) == 1 else "Not Completed"
        for value in predictions
    ]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = list(model.classes_)

        if 1 in classes:
            completed_index = classes.index(1)
            results["Completed_Probability"] = probabilities[:, completed_index]

    return results
