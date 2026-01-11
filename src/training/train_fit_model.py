import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

DATA_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("src/models/fit_model.joblib")

def main():
    df = pd.read_csv(DATA_PATH)

    X = df[["tfidf_sim", "sbert_sim", "overlap", "missing_count", "keyword_matches"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("\n✅ Classification Report:")
    print(classification_report(y_test, pred))

    print("✅ ROC AUC:", roc_auc_score(y_test, proba))

    print("\n✅ Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\n✅ Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
