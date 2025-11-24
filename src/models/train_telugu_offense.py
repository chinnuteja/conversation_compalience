# src/models/train_telugu_offense.py

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Paths & model name
DATA_DIR = Path("data/processed")

# Smaller multilingual model that supports Telugu and is faster on CPU
MODEL_NAME = "distilbert-base-multilingual-cased"

OUTPUT_DIR = Path("models/telugu_offense_v1_small")


def df_to_dataset(df: pd.DataFrame) -> Dataset:
    """
    Convert pandas DataFrame -> HF Dataset and drop any index columns like
    '__index_level_0__' that from_pandas may add.
    """
    ds = Dataset.from_pandas(df)
    cols_to_drop = [c for c in ds.column_names if c.startswith("__index_level_")]
    if cols_to_drop:
        ds = ds.remove_columns(cols_to_drop)
    return ds


def load_datasets() -> DatasetDict:
    """Load train/val/test CSVs and wrap into a DatasetDict (text + label only)."""
    train_df = pd.read_csv(DATA_DIR / "offense_train.csv")
    val_df = pd.read_csv(DATA_DIR / "offense_val.csv")
    test_df = pd.read_csv(DATA_DIR / "offense_test.csv")

    train_df = train_df[["text", "label"]]
    val_df = val_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    train_ds = df_to_dataset(train_df)
    val_ds = df_to_dataset(val_df)
    test_ds = df_to_dataset(test_df)

    return DatasetDict(
        train=train_ds,
        validation=val_ds,
        test=test_ds,
    )


def compute_metrics(eval_pred):
    """Compute accuracy / precision / recall / F1 for binary classification."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    # 1. Load datasets
    datasets = load_datasets()

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Tokenize (shorter max_length for speed)
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=32,
        )

    tokenized = datasets.map(tokenize_batch, batched=True)

    # Remove raw text column & set tensor format
    for split in tokenized.keys():
        if "text" in tokenized[split].column_names:
            tokenized[split] = tokenized[split].remove_columns(["text"])
        tokenized[split].set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )

    # 4. Load smaller model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # 5. Training arguments tuned for CPU
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,                  # 1 epoch for speed
        per_device_train_batch_size=32,      # larger batch to reduce steps
        per_device_eval_batch_size=64,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=100,
        do_eval=True,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    trainer.train()

    # 8. Final evaluation on the held-out test set
    test_metrics = trainer.evaluate(tokenized["test"])
    print("\n===== Test metrics =====")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    # 9. Save model + tokenizer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nSaved model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
