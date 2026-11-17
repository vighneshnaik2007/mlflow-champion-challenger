
import pandas as pd
import mlflow.pyfunc
from transformers import pipeline


class HFTextClassifier(mlflow.pyfunc.PythonModel):

    def __init__(self, model_id):
        self.model_id = model_id
        self.pipe = None

    def load_context(self, context):
        """Load the Hugging Face model when MLflow loads this model."""
        self.pipe = pipeline(
            task="text-classification",
            model=self.model_id,
            tokenizer=self.model_id,
            device=-1
        )

    def predict(self, context, model_input, params=None):
        """Accept text in a DataFrame and return sentiment predictions."""

        # Load the pipeline if it hasn't been loaded yet.
        if self.pipe is None:
            self.pipe = pipeline(
                task="text-classification",
                model=self.model_id,
                tokenizer=self.model_id,
                device=-1
            )

        texts = model_input["text"].astype(str).tolist()

        results = self.pipe(
            texts,
            batch_size=16,
            truncation=True,
            max_length=256
        )

        predictions = []
        scores = []

        for result in results:
            label = result["label"].upper()

            # Convert LABEL_0 / LABEL_1 to sentiment names.
            if label == "LABEL_0":
                label = "NEGATIVE"
            elif label == "LABEL_1":
                label = "POSITIVE"

            if label not in ["NEGATIVE", "POSITIVE"]:
                raise ValueError(
                    f"Unexpected sentiment label: {label}"
                )

            predictions.append(label.lower())
            scores.append(float(result["score"]))

        return pd.DataFrame({
            "prediction": predictions,
            "score": scores
        })
