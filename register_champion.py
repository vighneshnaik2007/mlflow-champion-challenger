
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-championchallenger"
REGISTERED_MODEL_NAME = "sentiment-classifier"

MODEL_A_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B_ID = "textattack/bert-base-uncased-SST-2"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError("MLflow experiment not found.")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=(
        "tags.mlflow.runName = 'compare_sentiment' "
        "and params.eval_samples = '200'"
    ),
    order_by=["start_time DESC"],
    max_results=1
)

if runs.empty:
    raise RuntimeError("Final 200-sample comparison run not found.")

comparison = runs.iloc[0]

winner = comparison["tags.winner"]
run_a = comparison["params.model_a_run_id"]
run_b = comparison["params.model_b_run_id"]

if winner not in [
    "A_sentiment_distilbert-sst2",
    "B_sentiment_textattack-bert-sst2"
]:
    raise RuntimeError(f"Unexpected winner value: {winner}")

if winner.startswith("A_"):
    champion_run_id = run_a
    challenger_run_id = run_b
    champion_model_id = MODEL_A_ID
    challenger_model_id = MODEL_B_ID
else:
    champion_run_id = run_b
    challenger_run_id = run_a
    champion_model_id = MODEL_B_ID
    challenger_model_id = MODEL_A_ID


def register_model(run_id, alias, model_id):
    model_uri = f"runs:/{run_id}/model"

    print(f"\nRegistering {alias}: {model_id}")

    version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=alias,
        version=version.version
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version.version,
        key="hf_model_id",
        value=model_id
    )

    print(f"Registered version: {version.version}")
    print(f"Alias assigned: {alias}")


register_model(
    champion_run_id,
    "champion",
    champion_model_id
)

register_model(
    challenger_run_id,
    "challenger",
    challenger_model_id
)

print("\nMODEL REGISTRATION COMPLETED")
print(f"Champion: {champion_model_id}")
print(f"Challenger: {challenger_model_id}")
