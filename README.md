
# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Student USN:** 1MS25CI143

## 1. Task and Models

- **Task:** Sentiment Classification
- **Dataset:** GLUE SST-2 Validation
- **Evaluation samples:** 200
- **Random seed:** 143
- **Device:** CPU
- **Primary metric:** Macro-F1

### Model A — Initial Champion

`distilbert-base-uncased-finetuned-sst-2-english`

### Model B — Challenger

`textattack/bert-base-uncased-SST-2`

## 2. Champion–Challenger Decision Rule

Model A (DistilBERT) is the initial Champion. Model B (TextAttack BERT) is the Challenger.

- Primary metric: Macro-F1. Higher is better.
- Model B replaces Model A only if its Macro-F1 is at least 0.01 higher.
- Latency guardrail: Model B must not have average inference latency greater than 5 times Model A's latency.
- If both conditions are not satisfied, Model A remains Champion.

This rule was established before the final 200-sample evaluation.

## 3. How to Run

Open PowerShell in the project root folder.

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project requirements:

```powershell
pip install -r requirements.txt
```

Start the MLflow server in a separate terminal:

```powershell
mlflow server --host 127.0.0.1 --port 5000
```

In the project terminal, run:

```powershell
python verify_models.py
python evaluate.py --samples 200
python register_champion.py
python predict_from_mlflow.py
```

Open the MLflow UI:

`http://127.0.0.1:5000`

## 4. Final Evaluation Results

- Dataset: GLUE SST-2 validation
- Evaluation samples: 200
- Random seed: 143
- Device: CPU

### Model Comparison

| Metric | Model A: DistilBERT | Model B: TextAttack BERT |
|---|---:|---:|
| Accuracy | 0.9050 | 0.9200 |
| Precision | 0.8824 | 0.8932 |
| Recall | 0.9278 | 0.9485 |
| F1 | 0.9045 | 0.9200 |
| Macro-F1 | 0.9050 | 0.9200 |
| Average latency (ms) | 12.7151 | 34.6893 |

### Champion Selection

**Selected Champion:** Model B — `textattack/bert-base-uncased-SST-2`

- Macro-F1 improvement: 0.0150
- Required improvement: 0.0100
- Latency ratio: 2.73x
- Maximum permitted latency ratio: 5x

Both selection conditions were satisfied.

### Decision Explanation

The Champion–Challenger evaluation was performed on 200 samples from the GLUE SST-2 validation dataset using the same inputs for both models. Model A, DistilBERT, achieved a Macro-F1 score of 0.9050, while Model B, TextAttack BERT, achieved 0.9200. The improvement in Macro-F1 was 0.0150, which exceeded the predefined minimum improvement of 0.0100.

Model B had an average inference latency of 34.6893 ms per sample, compared with 12.7151 ms for Model A. Its latency was approximately 2.73 times that of Model A, remaining within the maximum permitted ratio of 5 times.

Therefore, Model B satisfied both conditions in the decision rule and was selected as the new Champion. Although it was slower, its improvement in Macro-F1 met the predefined performance requirement without exceeding the latency limit. Model A was retained as the Challenger for comparison and future evaluation.

## 5. MLflow Model Registry

**Registered model name:** `sentiment-classifier`

- Champion alias: Version 1 — `textattack/bert-base-uncased-SST-2`
- Challenger alias: Version 2 — `distilbert-base-uncased-finetuned-sst-2-english`

Both models were registered under the same model name using MLflow aliases.

## 6. New Data Prediction

The Champion model was loaded using the MLflow alias:

`models:/sentiment-classifier@champion`

It was tested on 8 new text examples that were separate from the evaluation samples.

The script printed the model version and Hugging Face model ID and saved the predictions to:

`outputs/new_data_predictions.csv`

## 7. Limitations

- Evaluation was performed on 200 validation samples, not the entire dataset.
- Both models were evaluated using CPU.
- Inference latency may vary depending on hardware and system load.
- Results may differ on other datasets or text distributions.
- A larger evaluation sample may provide a more reliable estimate of model performance.

## 8. AI Use Declaration

ChatGPT was used for assignment guidance, project planning, code drafting, debugging assistance, and documentation. The models were evaluated locally, and the reported metrics and predictions were obtained from the actual MLflow experiment runs. The student is responsible for reviewing, understanding, and explaining the implementation and results.
