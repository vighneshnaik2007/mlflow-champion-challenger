
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-championchallenger"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError("MLflow experiment not found.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'compare_sentiment'",
    order_by=["attribute.start_time DESC"],
    max_results=1
)

if not runs:
    raise RuntimeError("Comparison run not found.")

run_id = runs[0].info.run_id

client.log_artifact(
    run_id,
    "README.md"
)

print("README.md successfully added to comparison run artifacts.")
print("Run ID:", run_id)
