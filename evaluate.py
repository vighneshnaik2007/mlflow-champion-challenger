
import argparse
import gc
import json
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import torch
import transformers

from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from model_wrapper import HFTextClassifier


# ---------------- PROJECT SETTINGS ----------------

USN = "1MS25CI143"
SEED = 143
DEVICE = "cpu"
SAMPLE_COUNT_DEFAULT = 10

MODEL_A = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B = "textattack/bert-base-uncased-SST-2"

EXPERIMENT_NAME = "hf-sentiment-championchallenger"
TRACKING_URI = "http://127.0.0.1:5000"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------- LABEL CONVERSION ----------------

def normalize_label(label):
    label = str(label).upper()

    if label in ["POSITIVE", "LABEL_1"]:
        return "positive"

    if label in ["NEGATIVE", "LABEL_0"]:
        return "negative"

    raise ValueError(f"Unexpected model label: {label}")


# ---------------- LOAD DATASET ----------------

def load_evaluation_data(sample_count):

    print("\nLoading GLUE SST-2 validation dataset...")

    dataset = load_dataset(
    "nyu-mll/glue",
    "sst2",
    split="validation"
)

    dataset = dataset.shuffle(
        seed=SEED
    ).select(range(sample_count))

    df = pd.DataFrame({
        "text": dataset["sentence"],
        "reference": [
            "positive" if label == 1 else "negative"
            for label in dataset["label"]
        ]
    })

    df.to_csv(
        DATA_DIR / "eval_inputs.csv",
        index=False
    )

    print("Evaluation samples:", len(df))

    return df


# ---------------- CALCULATE METRICS ----------------

def calculate_metrics(references, predictions):

    return {
        "accuracy": float(
            accuracy_score(references, predictions)
        ),

        "precision": float(
            precision_score(
                references,
                predictions,
                pos_label="positive",
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                references,
                predictions,
                pos_label="positive",
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                references,
                predictions,
                pos_label="positive",
                zero_division=0
            )
        ),

        "macro_f1": float(
            f1_score(
                references,
                predictions,
                average="macro",
                zero_division=0
            )
        )
    }


# ---------------- CONFUSION MATRIX ----------------

def save_confusion_matrix(references, predictions, path, title):

    matrix = confusion_matrix(
        references,
        predictions,
        labels=["negative", "positive"]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )

    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------- EVALUATE ONE MODEL ----------------

def evaluate_model(model_id, run_name, role, df, model_key):

    print("\n" + "=" * 60)
    print("Evaluating:", model_id)
    print("=" * 60)

    classifier = pipeline(
        task="text-classification",
        model=model_id,
        tokenizer=model_id,
        device=-1
    )

    texts = df["text"].tolist()
    references = df["reference"].tolist()

    # Run inference and measure inference time.
    start_time = time.perf_counter()

    raw_results = classifier(
        texts,
        batch_size=16,
        truncation=True,
        max_length=256
    )

    total_inference_s = time.perf_counter() - start_time

    avg_latency_ms = (
        total_inference_s * 1000 / len(texts)
    )

    predictions = [
        normalize_label(result["label"])
        for result in raw_results
    ]

    scores = [
        float(result["score"])
        for result in raw_results
    ]

    metrics = calculate_metrics(
        references,
        predictions
    )

    metrics["avg_latency_ms"] = float(avg_latency_ms)
    metrics["total_inference_s"] = float(total_inference_s)

    # Save prediction CSV.
    prediction_df = pd.DataFrame({
        "input": texts,
        "reference": references,
        "prediction": predictions,
        "confidence": scores
    })

    prediction_path = (
        OUTPUT_DIR / f"predictions_{model_key}.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # Save metrics JSON.
    metrics_path = (
        OUTPUT_DIR / f"metrics_{model_key}.json"
    )

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Save confusion matrix.
    confusion_path = (
        OUTPUT_DIR / f"confusion_matrix_{model_key}.png"
    )

    save_confusion_matrix(
        references,
        predictions,
        confusion_path,
        run_name
    )

    # Log experiment to MLflow.
    with mlflow.start_run(run_name=run_name) as run:

        mlflow.log_params({
            "hf_model_id": model_id,
            "task": "sentiment",
            "role": role,
            "max_length": 256,
            "batch_size": 16,
            "eval_samples": len(df),
            "seed": SEED,
            "device": DEVICE
        })

        mlflow.log_metrics(metrics)

        mlflow.set_tags({
            "student_usn": USN,
            "dataset_name": "GLUE-SST2-validation",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "mlflow_version": mlflow.__version__,
            "scikit_learn_version": sklearn.__version__
        })

        mlflow.log_artifact(str(prediction_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(confusion_path))

        # Log the model as an MLflow PyFunc model.
        mlflow.pyfunc.log_model(
            name="model",
            python_model=HFTextClassifier(model_id),
            code_paths=[str(ROOT / "model_wrapper.py")],
            input_example=pd.DataFrame({
                "text": texts[:2]
            })
        )

        run_id = run.info.run_id

    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.6f}")

    del classifier
    gc.collect()

    return {
        "run_id": run_id,
        "predictions": predictions,
        "metrics": metrics
    }


# ---------------- MAIN EXPERIMENT ----------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=SAMPLE_COUNT_DEFAULT
    )

    args = parser.parse_args()

    if args.samples < 1 or args.samples > 872:
        raise ValueError(
            "Sample count must be between 1 and 872."
        )

    # Load the same evaluation inputs for both models.
    df = load_evaluation_data(args.samples)

    # Model A: initial Champion.
    result_a = evaluate_model(
        MODEL_A,
        "A_sentiment_distilbert-sst2",
        "champion_candidate",
        df,
        "A"
    )

    # Model B: Challenger.
    result_b = evaluate_model(
        MODEL_B,
        "B_sentiment_textattack-bert-sst2",
        "challenger_candidate",
        df,
        "B"
    )

    # Calculate Macro-F1 difference.
    delta = (
        result_b["metrics"]["macro_f1"]
        - result_a["metrics"]["macro_f1"]
    )

    # Check latency guardrail.
    latency_guardrail = (
        result_b["metrics"]["avg_latency_ms"]
        <= 5 * result_a["metrics"]["avg_latency_ms"]
    )

    # Apply the prewritten decision rule.
    if delta >= 0.01 and latency_guardrail:
        winner = "B_sentiment_textattack-bert-sst2"
    else:
        winner = "A_sentiment_distilbert-sst2"

    # Save metric comparison chart.
    comparison_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Macro-F1"
        ],
        "Model A": [
            result_a["metrics"]["accuracy"],
            result_a["metrics"]["precision"],
            result_a["metrics"]["recall"],
            result_a["metrics"]["f1"],
            result_a["metrics"]["macro_f1"]
        ],
        "Model B": [
            result_b["metrics"]["accuracy"],
            result_b["metrics"]["precision"],
            result_b["metrics"]["recall"],
            result_b["metrics"]["f1"],
            result_b["metrics"]["macro_f1"]
        ]
    })

    comparison_path = (
        OUTPUT_DIR / "metric_comparison.png"
    )

    comparison_plot = comparison_df.set_index("Metric")

    comparison_plot.plot(
        kind="bar",
        figsize=(9, 5)
    )

    plt.title("Model A vs Model B - Sentiment Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    # Save actual disagreements.
    disagreement_df = pd.DataFrame({
        "input": df["text"],
        "reference": df["reference"],
        "model_a_prediction": result_a["predictions"],
        "model_b_prediction": result_b["predictions"]
    })

    disagreement_df = disagreement_df[
        disagreement_df["model_a_prediction"]
        != disagreement_df["model_b_prediction"]
    ].head(5).copy()

    disagreement_df["comment"] = (
        "Review the wording and context to understand "
        "why the two models disagree."
    )

    disagreement_path = (
        OUTPUT_DIR / "disagreements.csv"
    )

    disagreement_df.to_csv(
        disagreement_path,
        index=False
    )

    # Save combined summary.
    summary = {
        "student_usn": USN,
        "dataset": "GLUE-SST2-validation",
        "samples": len(df),
        "seed": SEED,
        "device": DEVICE,
        "model_a": result_a["metrics"],
        "model_b": result_b["metrics"],
        "delta_macro_f1": float(delta),
        "latency_guardrail_passed": bool(latency_guardrail),
        "winner": winner
    }

    summary_path = (
        OUTPUT_DIR / "metrics_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # Create comparison MLflow run.
    with mlflow.start_run(
        run_name="compare_sentiment"
    ):

        mlflow.log_params({
            "task": "sentiment",
            "primary_metric": "macro_f1",
            "required_margin": 0.01,
            "latency_limit_multiplier": 5,
            "eval_samples": len(df),
            "seed": SEED,
            "model_a_run_id": result_a["run_id"],
            "model_b_run_id": result_b["run_id"]
        })

        mlflow.log_metric(
            "delta_macro_f1",
            float(delta)
        )

        mlflow.set_tags({
            "student_usn": USN,
            "dataset_name": "GLUE-SST2-validation",
            "winner": winner
        })

        mlflow.log_artifact(
            str(DATA_DIR / "eval_inputs.csv")
        )

        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(comparison_path))
        mlflow.log_artifact(str(disagreement_path))

    # Print final results.
    print("\n" + "=" * 60)
    print("CHAMPION-CHALLENGER COMPARISON COMPLETE")
    print("=" * 60)

    print(
        "Model A Macro-F1:",
        result_a["metrics"]["macro_f1"]
    )

    print(
        "Model B Macro-F1:",
        result_b["metrics"]["macro_f1"]
    )

    print("Macro-F1 difference:", delta)
    print("Latency guardrail passed:", latency_guardrail)
    print("Selected Champion:", winner)
    print("Disagreement examples:", len(disagreement_df))

    print("\nOutputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
