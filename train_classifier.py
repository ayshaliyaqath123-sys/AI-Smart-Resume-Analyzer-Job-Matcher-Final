"""
train_classifier.py
Trains a TF-IDF + RandomForest pipeline to predict a resume's job category
(Data Scientist, Software Engineer, etc.) and saves it to models/.

Run this once before starting the Flask app:
    python train_classifier.py

The sample dataset in data/sample_resumes.csv is tiny and synthetic - it's
only here so the project works out of the box. For real accuracy, download
a proper labeled dataset from Kaggle (see README "Dataset" section) and
point CSV_PATH at it. Same code, bigger data, better model.
"""
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_resumes.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "resume_classifier.joblib")


def main():
    print(f"Loading training data from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["resume_text", "category"])

    X = df["resume_text"]
    y = df["category"]

    # Dataset is tiny, so we skip a held-out test split when there isn't
    # enough data per class; otherwise we report a quick accuracy check.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=2000)),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])

    if len(df) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        print("\nHold-out evaluation:")
        print(classification_report(y_test, preds, zero_division=0))
        # Refit on all data before saving so the deployed model uses everything
        pipeline.fit(X, y)
    else:
        pipeline.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
