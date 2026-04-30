"""Load the Production model from MLflow registry and predict."""

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd

# ── 1. Connect to DagsHub MLflow ──────────────────────────────────
dagshub.init(
    repo_owner="omar.sameh.shamakh",
    repo_name="lab0-titanic-pipeline",
    mlflow=True
)

# ── 2. Load the model using the production alias ───────────────────
MODEL_NAME = "titanic-random_forest"
model_uri = f"models:/{MODEL_NAME}@production"

print(f"📦 Loading model: {model_uri}")
model = mlflow.sklearn.load_model(model_uri)
print("✅ Model loaded successfully!")

# ── 3. Create sample passengers to predict ────────────────────────
sample_passengers = pd.DataFrame({
    "Pclass":      [1,        3,       2],
    "Sex":         ["female", "male",  "female"],
    "Age":         [29.0,     22.0,    35.0],
    "SibSp":       [0,        1,       1],
    "Parch":       [0,        0,       0],
    "Fare":        [211.3,    7.25,    26.0],
    "Embarked":    ["S",      "S",     "C"],
    "Family_size": [1,        2,       2],
    "Alone":       [1,        0,       0],
    "Initial":     ["Mrs",    "Mr",    "Mrs"],
    "Fare_cat":    [3,        0,       1],
    "Age_band":    [2,        2,       3],
})

# ── 4. Predict ────────────────────────────────────────────────────
predictions = model.predict(sample_passengers)
probabilities = model.predict_proba(sample_passengers)

print("\n🚢 Titanic Survival Predictions:")
print("-" * 45)
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    survived = "✅ SURVIVED" if pred == 1 else "❌ DID NOT SURVIVE"
    confidence = prob[pred] * 100
    print(f"Passenger {i+1}: {survived} ({confidence:.1f}% confidence)")