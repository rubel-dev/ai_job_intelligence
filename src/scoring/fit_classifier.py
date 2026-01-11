import joblib
import pandas as pd
from pathlib import Path
import shap

from .features import FEATURE_NAMES

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "fit_model.joblib"

class FitClassifier:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Train it first.")
        self.model = joblib.load(MODEL_PATH)

        # SHAP explainer (LogReg is linear -> fast)
        self.explainer = shap.LinearExplainer(self.model, pd.DataFrame([[0]*len(FEATURE_NAMES)], columns=FEATURE_NAMES))

    def predict_with_explain(self, feature_values: list):
        X = pd.DataFrame([feature_values], columns=FEATURE_NAMES)

        proba = float(self.model.predict_proba(X)[0][1])

        shap_values = self.explainer.shap_values(X)[0]
        shap_map = {FEATURE_NAMES[i]: float(shap_values[i]) for i in range(len(FEATURE_NAMES))}

        # Sort by strongest impact
        shap_sorted = dict(sorted(shap_map.items(), key=lambda x: abs(x[1]), reverse=True))

        return proba, shap_sorted
