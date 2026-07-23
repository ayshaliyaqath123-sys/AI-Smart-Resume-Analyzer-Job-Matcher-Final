"""
classifier.py
Loads the trained resume-category classifier (TF-IDF + RandomForest,
trained by train_classifier.py) and predicts a category for new resume text.
"""
import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "resume_classifier.joblib")

_model = None


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Trained model not found. Run `python train_classifier.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_category(resume_text: str) -> dict:
    """Returns predicted category plus the model's confidence."""
    model = _load_model()
    proba = model.predict_proba([resume_text])[0]
    classes = model.classes_
    best_idx = proba.argmax()

    return {
        "category": classes[best_idx],
        "confidence": round(float(proba[best_idx]) * 100, 1),
    }


def model_available() -> bool:
    return os.path.exists(MODEL_PATH)
