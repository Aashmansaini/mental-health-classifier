import pandas as pd
import numpy as np
import re
import time
import torch
import joblib
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


# Check if a GPU is available and use it, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load the dataset and drop any rows with missing values
df = pd.read_csv("data/Combined Data.csv")[["statement", "status"]].dropna()
df["statement"] = df["statement"].apply(clean_text)
print(f"Loaded {len(df):,} rows across {df['status'].nunique()} classes")

# Encode the text labels into integers for training
le = LabelEncoder()
df["label"] = le.fit_transform(df["status"])
print(f"Classes: {list(le.classes_)}")

# Split into 80% training and 20% test, keeping class proportions balanced
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"Train size: {len(train_df):,} | Test size: {len(test_df):,}")


# Load the RoBERTa tokenizer
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded.")


def tokenize(batch):
    return tokenizer(
        batch["statement"],
        truncation=True,
        max_length=128
    )


# Convert dataframes to HuggingFace datasets and tokenize
train_dataset = Dataset.from_pandas(train_df[["statement", "label"]].reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df[["statement", "label"]].reset_index(drop=True))

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("Tokenization complete.")


# Load RoBERTa with a classification head for our 7 mental health categories
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(le.classes_),
    id2label={i: label for i, label in enumerate(le.classes_)},
    label2id={label: i for i, label in enumerate(le.classes_)}
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


training_args = TrainingArguments(
    output_dir="roberta_checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    logging_steps=100,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("Starting training...")
start = time.time()
trainer.train()
print(f"Training complete in {(time.time() - start) / 60:.1f} minutes")


# Evaluate on the held-out test set
print("\nEvaluating on test set...")
preds_output = trainer.predict(test_dataset)
y_pred = np.argmax(preds_output.predictions, axis=-1)
y_true = test_df["label"].values

accuracy = accuracy_score(y_true, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_true, y_pred, target_names=le.classes_))


# Save the model, tokenizer and label encoder for use in the Flask app
model.save_pretrained("roberta_model")
tokenizer.save_pretrained("roberta_model")
joblib.dump(le, "label_encoder.pkl")
print("Model saved to roberta_model/")
