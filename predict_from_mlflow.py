
import mlflow
import pandas as pd
from pathlib import Path
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "sentiment-classifier"

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "data" / "new_data.csv"
OUTPUT_FILE = ROOT / "outputs" / "new_data_predictions.csv"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()

# Verify the Champion alias
version = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
)

print("Champion model version:", version.version)
print("Champion Hugging Face ID:", version.tags.get("hf_model_id"))

# Load the Champion model through MLflow
model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}@champion"
)

# Load new input examples
new_data = pd.read_csv(INPUT_FILE)

# Generate predictions
predictions = model.predict(new_data[["text"]])

new_data["prediction"] = predictions["prediction"]
new_data["confidence"] = predictions["score"]

# Save predictions
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
new_data.to_csv(OUTPUT_FILE, index=False)

print("\nPredictions:")
print(new_data.to_string(index=False))

print(f"\nSaved predictions to: {OUTPUT_FILE}")
