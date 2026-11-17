
from transformers import pipeline, AutoConfig
import gc

models = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "textattack/bert-base-uncased-SST-2"
]

test_sentences = [
    "This movie was absolutely wonderful!",
    "This movie was terrible and boring."
]

for model_id in models:
    print("\n" + "=" * 60)
    print("MODEL:", model_id)

    config = AutoConfig.from_pretrained(model_id)
    print("LABEL MAPPING:", config.id2label)

    classifier = pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        device=-1
    )

    results = classifier(test_sentences)

    for sentence, result in zip(test_sentences, results):
        print("\nText:", sentence)
        print("Prediction:", result)

    del classifier
    gc.collect()

print("\nMODEL VERIFICATION COMPLETED")
